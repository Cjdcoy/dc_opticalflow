from __future__ import print_function
import os, sys
from scipy import misc
import caffe
import tempfile
from math import ceil
from threading import Thread
import socket
import struct
import cv2
import numpy as np
import time
import argparse
import cPickle as pickle


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
    return writeFlow(blob)


def makeColorwheel():
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

def computeColor(u, v):
    colorwheel = makeColorwheel();
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


def computeImg(flow):
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
    #fix unknown flow
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
    #print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = computeColor(u, v)
    return img


def writeFlow(flow):
    return computeImg(flow.astype(np.float32))


class Receiving(Thread):

    def __init__(self):
        #Thread.__init__(self)
        self.chrono = time.time()

    def receive_image(self, sc):
        len_str = sc.recv(4)
        size = struct.unpack('!i', len_str)[0]

        blob = b''
        while size > 0:
            if size >= 4096:
                data = sc.recv(4096)
            else:
                data = sc.recv(size)

            if not data:
                break

            size -= len(data)
            blob += data
        if sys.version_info.major < 3:
            unserialized_blob = pickle.loads(blob)
        else:
            unserialized_blob = pickle.loads(blob, encoding='bytes')
        return unserialized_blob

    def send_image(self, sc, flow):
        serialized_data = pickle.dumps(flow, protocol=2)
        sc.send(struct.pack('!i', len(serialized_data)))
        sc.send(serialized_data)

    def fps_counter(self, nb_loop):
        if time.time() - self.chrono > 1:
            fps = nb_loop / (time.time() - self.chrono)
            self.chrono = time.time()
            print(fps)
            return 0
        return nb_loop

    def __listen_client(self, sc, args):
        first_loop = True
        running = True
        nb_loop = 0
        while running:
            img = self.receive_image(sc)
            if first_loop:
                first_loop = False
                img_past = img
            nb_loop = self.fps_counter(nb_loop)
            nb_loop += 1
            flow = opticalflow_NN(img_past, img, args)
            self.send_image(sc, flow)
            img_past = img

    def run(self, args):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.ip, args.port))
        s.listen(1)
        try:
            while True:
                try:
                    sc, info = s.accept()
                    self.__listen_client(sc, args)

                except struct.error as e:
                    cv2.destroyAllWindows()

        except Exception as e:
            print(e)
            pass

        finally:
            print("Closing socket and exit")
            s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", help="default = any ip is allowed else only the ip specified is allowed", type=str, default="")
    parser.add_argument("-p", "--port", type=int, default=10000)
    parser.add_argument("-c", "--caffemodel", help='path to model', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5")
    parser.add_argument("-d", "--deployproto",  help='path to deploy prototxt template', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template")
    args = parser.parse_args()

    Receiving().run(args)
