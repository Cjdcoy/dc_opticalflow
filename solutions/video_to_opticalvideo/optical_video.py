from __future__ import print_function
import cv2
import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import time
from datetime import datetime, timedelta

#python2 optical_video.py -c ../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5 -d ../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template -pre 0 -s test
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--caffemodel", help='path to model', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5")
parser.add_argument("-d", "--deployproto",  help='path to deploy prototxt template', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template")

parser.add_argument('--width', help='set width, default 320', default=320, type=int)
parser.add_argument('--height', help='set hight, default 240', default=240, type=int)
parser.add_argument("-pre", "--preview", help="[-1] no preview [0] image (default), [1] image+fps, [2] print+image+fps, [3] print+image",
                    type=int, default=0, choices=[-1, 0, 1, 2, 3])

parser.add_argument("-l", "--list", help="file containing image/video list. Format: \"path\\npath...\"", type=str, default="video_list_example")
parser.add_argument("-s", "--save", help="save flow under [string].avi or save videos/images in folder [string] (empty/default: no save)", type=str, default="")
parser.add_argument("-f", "--fps", help="choose how many fps will have the video you receive from the server", type=int, default=20)

parser.add_argument("-e", "--estimation", help="[0] no computing estimation [1] simple estimate [2] complete estimation", type=int, default=0, choices=[0, 1, 2])


parser.add_argument('--verbose', help='whether to output all caffe logging',
                    action='store_true')

args = parser.parse_args()

if not args.verbose:
    caffe.set_logging_disabled()
caffe.set_device(0)
caffe.set_mode_gpu()
check = False
tmp = None
net = None
def opticalflow_NN(img0, img1, args):
    global check
    global caffe
    global tmp
    global net

    num_blobs = 2
    input_data = []
    if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

    width = input_data[0].shape[3]
    height = input_data[0].shape[2]
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height
    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)
    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

    if check == False:
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)
        proto = open(args.deployproto).readlines()
        for line in proto:
            for key, value in vars.items():
                tag = "$%s$" % key
                line = line.replace(tag, str(value))
            tmp.write(line)
        tmp.flush()
        net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)
    check = True
    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
#
# There is some non-deterministic nan-bug in caffe
# it seems to be a race-condition
#
    i = 1
    while i<=5:
        i+=1
        net.forward(**input_dict)
        containsNaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()
            if has_nan:
                containsNaN = True
        if not containsNaN:
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')
    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
    return write_flow(blob)


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)


def read_flow(file):
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    print(flow)
    return flow


def make_color_wheel():
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
    colorwheel = np.zeros([ncols, 3]) # r g b
    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY
    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255;
    col += YG;
    #GC
    colorwheel[col:GC+col, 1]= 255
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC;
    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB;
    #BM
    colorwheel[col:BM+col, 2]= 255
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM;
    #MR
    colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return  colorwheel


def compute_color(u, v):
    colorwheel = make_color_wheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)     # 1, 2, ..., ncols
    k1 = k0+1;
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)


def compute_image(flow):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    u = flow[: , : , 0]
    v = flow[: , : , 1]
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
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
    maxrad = max([maxrad, np.amax(rad)])
    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = compute_color(u, v)
    return img


def write_flow(flow):
    return compute_image(flow.astype(np.float32))


class FpsMetter(object):
    def __init__(self):
        self.chrono = time.time()
        self.fps = 0
        self.first_loop = True
        self.init_finished = False

    def get_fps(self, nb_loop):
        if time.time() - self.chrono > 1:
            if not self.first_loop:
                self.init_finished = True
            self.first_loop = False
            self.fps = nb_loop / (time.time() - self.chrono)
            self.chrono = time.time()
            if args.preview > 1:
                print(self.fps)
            return 0
        nb_loop += 1
        return nb_loop


class OpticalVideoList(object):
    def __init__(self):
        self.fpsMetter = FpsMetter()
        self.video_list = open(args.list, 'r').readlines()
        self.cursor = 0
        self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
        self.width = args.width
        self.height = args.height
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.videoFps = args.fps
        self.estimation = False
        if args.estimation > 0:
            self.estimation = True
        if len(args.save) > 0:
            if not os.path.exists(args.save):
                os.makedirs(args.save)
            self.out = cv2.VideoWriter(args.save + "/" + str(self.cursor) + ".avi", self.fourcc, args.fps, (args.width, args.height))

    def __del__(self):
        self.video.release()

    def load_new_video(self, save):
        self.cursor += 1
        if self.cursor < len(self.video_list):
            self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
            if save:
                self.out.release()
                self.out = cv2.VideoWriter(args.save + "/" + str(self.cursor) + ".avi", self.fourcc, self.videoFps, (self.width,  self.height))

    def get_frame(self, save):
        success, image = self.video.read()
        if success:
            image = cv2.resize(image, (self.width, self.height))
        else:
            self.load_new_video(save)
            success, image = self.video.read()
            if success:
                image = cv2.resize(image, (self.width, self.height))
        return success, image

    def fast_get_frame(self, save):
        success, image = self.video.read()
        if success:
            return success
        else:
            self.load_new_video(save)
            print("loaded video number", self.cursor)
            success, image = self.video.read()
        return success

    def save_flow(self, flow):
        self.out.write(flow)

    def estimate_compute_time(self, fps):
        self.estimation = False
        total_nb_frame = 0
        print("Calculating compute time...\nEstimated FPS: " + "{:1.2f}".format(
            fps) + "\n")
        for i in range(0, len(self.video_list)):
            cap = cv2.VideoCapture(self.video_list[i].replace("\n", ""))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.estimation == 2:
                print("video " + str(i) + ": " + str(
                    frames) + " frames (" + "{:1.2f}".format(
                    frames / fps) + " seconds)")
            total_nb_frame += frames
        print("\nThere are " + str(total_nb_frame) + " frames to compute")
        print("Estimated compute time (day, hour, min, sec):")
        sec = timedelta(seconds=total_nb_frame / fps)
        d = datetime(1, 1, 1) + sec
        print("{:02d}".format(d.day - 1) + ":" + "{:02d}".format(
            d.hour) + ":" + "{:02d}".format(d.minute) + ":" + "{:02d}".format(
            d.second))

    def preview(self, nb_loop, flow):
        nb_loop = self.fpsMetter.get_fps(nb_loop)
        if self.fpsMetter.init_finished and self.estimation: #does not estimate if the solution takes more than 1 second per image
            self.estimate_compute_time(self.fpsMetter.fps)
        if args.preview > 0 and args.preview != 3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flow, "fps: " + "{:1.2f}".format(self.fpsMetter.fps), (10, 20),
                        font, 0.5, (10, 10, 10), 2,
                        cv2.LINE_AA)
        if args.preview > -1:
            cv2.imshow('opticalflow received', flow)
            if cv2.waitKey(1) & 0xFF == 27:
                return -1
        return nb_loop

    def run_rendering(self):
        save = False
        if len(args.save) > 0:
            save = True
        ret_val, prev_img = self.get_frame(save)
        nb_loop = 0
        while True:
            ret_val, actual_img = self.get_frame(save)
            if ret_val:
                flow_img = opticalflow_NN(prev_img, actual_img, args)
                if len(args.save) > 0:
                    self.save_flow(flow_img)                #save video
                nb_loop = self.preview(nb_loop, flow_img)   #preview / estimated
                if nb_loop == -1:
                    break
                prev_img = actual_img
            else:
                break
        self.video.release()
        cv2.destroyAllWindows()


def main():
    OpticalVideoList().run_rendering()


if __name__ == '__main__':
    main()