from __future__ import print_function
import os, sys, numpy as np
from scipy import misc
import caffe
import tempfile
from math import ceil

#Computeimage is a generic class that allows us easily change the algorithm within the testing environment
#you can load the properties you want placing a file .csv
#csv format : type,value. Type can either be INT or STR
#class variables will be initiated from the csv in their declaration order

caffe.set_logging_disabled()
caffe.set_device(0)
caffe.set_mode_gpu()

class ComputeImage(object):
    def __init__(self, values_to_load='compute_image.csv'):
        caffe.set_logging_disabled()
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.check = False
        self.tmp = None
        self.net = None
        self.colorwheel = self.make_color_wheel()
        if os.path.isfile(values_to_load):
            self.init_class_from_csv(values_to_load)

    def init_class_from_csv(self, values_to_load):
        content = open(values_to_load, 'r').read().split(',')
        it = 0

        for dir in self.__dir__():
            if dir in self.__dict__ and it < len(content):
                if content[it] == 'int':
                    self.__dict__[dir] = int(content[it + 1])
                if content[it] == 'str':
                    self.__dict__[dir] = str(content[it + 1])
                it += 2

    def run(self, img0, img1, args):
        global caffe

        num_blobs = 2
        input_data = []
        if len(img0.shape) < 3:
            input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:
            input_data.append(
                img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :,
                :])
        if len(img1.shape) < 3:
            input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:
            input_data.append(
                img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :,
                :])

        width = input_data[0].shape[3]
        height = input_data[0].shape[2]
        vars = {}
        vars['TARGET_WIDTH'] = width
        vars['TARGET_HEIGHT'] = height
        divisor = 64.
        vars['ADAPTED_WIDTH'] = int(ceil(width / divisor) * divisor)
        vars['ADAPTED_HEIGHT'] = int(ceil(height / divisor) * divisor)
        vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
        vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

        if self.check == False:
            self.tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)
            proto = open(args.deployproto).readlines()
            for line in proto:
                for key, value in vars.items():
                    tag = "$%s$" % key
                    line = line.replace(tag, str(value))
                self.tmp.write(line)
            self.tmp.flush()
            self.net = caffe.Net(self.tmp.name, args.caffemodel, caffe.TEST)
        self.check = True
        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[self.net.inputs[blob_idx]] = input_data[blob_idx]
        #
        # There is some non-deterministic nan-bug in caffe
        # it seems to be a race-condition
        #
        i = 1
        while i <= 5:
            i += 1
            self.net.forward(**input_dict)
            containsNaN = False
            for name in self.net.blobs:
                blob = self.net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()
                if has_nan:
                    containsNaN = True
            if not containsNaN:
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')
        blob = np.squeeze(self.net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        return self.write_flow(blob)

    def write_flow(self, flow):
        return self.compute_image(flow.astype(np.float32))

    def compute_image(self, flow):
        eps = sys.float_info.epsilon
        UNKNOWN_FLOW_THRESH = 1e9
        UNKNOWN_FLOW = 1e10
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        maxu = -999
        maxv = -999
        minu = 999
        minv = 999
        maxrad = -1

        greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
        greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
        u[greater_u] = 0
        u[greater_v] = 0
        v[greater_u] = 0
        v[greater_v] = 0

        maxu = max([maxu, np.amax(u)])
        minu = min([minu, np.amin(u)])
        maxv = max([maxv, np.amax(v)])
        minv = min([minv, np.amin(v)])
        rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
        maxrad = max([maxrad, np.amax(rad)])
        u = u / (maxrad + eps)
        v = v / (maxrad + eps)
        img = self.compute_color(u, v)
        return img

    def compute_color(self, u, v):
        nan_u = np.isnan(u)
        nan_v = np.isnan(v)
        nan_u = np.where(nan_u)
        nan_v = np.where(nan_v)

        u[nan_u] = 0
        u[nan_v] = 0
        v[nan_u] = 0
        v[nan_v] = 0

        ncols = self.colorwheel.shape[0]
        radius = np.sqrt(u ** 2 + v ** 2)
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
        k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
        k1 = k0 + 1;
        k1[k1 == ncols] = 0
        f = fk - k0

        img = np.empty([k1.shape[0], k1.shape[1], 3])
        ncolors = self.colorwheel.shape[1]
        for i in range(ncolors):
            self.tmp = self.colorwheel[:, i]
            col0 = self.tmp[k0] / 255
            col1 = self.tmp[k1] / 255
            col = (1 - f) * col0 + f * col1
            idx = radius <= 1
            col[idx] = 1 - radius[idx] * (
                        1 - col[idx])  # increase saturation with radius
            col[~idx] *= 0.75  # out of range
            img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)
        return img.astype(np.uint8)

    def make_color_wheel(self):
        #  color encoding scheme
        #   adapted from the color circle idea described at
        #   http://members.shaw.ca/quadibloc/other/colint.htm
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros([ncols, 3])  # r g b
        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
        col += RY
        # YG
        colorwheel[col:YG + col, 0] = 255 - np.floor(
            255 * np.arange(0, YG, 1) / YG)
        colorwheel[col:YG + col, 1] = 255;
        col += YG;
        # GC
        colorwheel[col:GC + col, 1] = 255
        colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
        col += GC;
        # CB
        colorwheel[col:CB + col, 1] = 255 - np.floor(
            255 * np.arange(0, CB, 1) / CB)
        colorwheel[col:CB + col, 2] = 255
        col += CB;
        # BM
        colorwheel[col:BM + col, 2] = 255
        colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
        col += BM;
        # MR
        colorwheel[col:MR + col, 2] = 255 - np.floor(
            255 * np.arange(0, MR, 1) / MR)
        colorwheel[col:MR + col, 0] = 255
        return colorwheel

    def readFlow(self, name):
        if name.endswith('.pfm') or name.endswith('.PFM'):
            return readPFM(name)[0][:, :, 0:2]
        f = open(name, 'rb')
        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape(
            (height, width, 2))
        return flow.astype(np.float32)

    def read_flow(self, file):
        TAG_FLOAT = 202021.25
        assert type(file) is str, "file is not str %r" % str(file)
        assert os.path.isfile(file) is True, "file does not exist %r" % str(
            file)
        assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
        f = open(file, 'rb')
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
        data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        flow = np.resize(data, (int(h), int(w), 2))
        f.close()
        return flow