from __future__ import print_function
import cv2
import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import time
import PIL

#USAGE
#python2 realtime_optical.py ../../models/FlowNet2-KITTI/FlowNet2-KITTI_weights.caffemodel.h5 ../../models/FlowNet2-KITTI/FlowNet2-KITTI_deploy.prototxt.template
#python2 realtime_optical.py ../../models/FlowNet2-Sintel/FlowNet2-CSS-Sintel_weights.caffemodel.h5 ../../models/FlowNet2-Sintel/FlowNet2-CSS-Sintel_deploy.prototxt.template
#python2 realtime_optical.py ../../models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5 ../../models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template

#python2 realtime_optical.py ../../models/FlowNet2/FlowNet2_weights.caffemodel.h5 ../../models/FlowNet2/FlowNet2_deploy.prototxt.template

#python2 realtime_optical.py ../../models/FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_weights.caffemodel.h5 ../../models/FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_deploy.prototxt.template
#python2 realtime_optical.py ../../models/FlowNet2-css-ft-sd/FlowNet2-css-ft-sd_weights.caffemodel.h5 ../../models/FlowNet2-css-ft-sd/FlowNet2-css-ft-sd_deploy.prototxt.template

#yes "python2 realtime_optical.py ../../models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5 ../../models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template &" | head -n 4 | bash

parser = argparse.ArgumentParser()

parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
#parser.add_argument('width', help='set width, default 320', default=320, type=int)
#parser.add_argument('height', help='set hight, default 240', default=240, type=int)
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
                #print('blob %s contains nan' % name)
                containsNaN = True

        if not containsNaN:
            #print('Succeeded.')
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')


    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
    return writeFlow(blob)

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

TAG_FLOAT = 202021.25

def read_flow(file):
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

start_time = time.time()
def show_webcam(args, mirror=False):
    global start_time
    cam = cv2.VideoCapture(0) #/home/cjdcoy/Downloads/Office/v10.avi
    width = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) / 2
    height = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) / 2
    print(width, height)
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    ret_val, prev_img = cam.read()
    prev_img = cv2.resize(prev_img, (int(width), int(height)))
    i = 0
    x = 1
    fps = 0
    while True:
        i += 1
        ret_val, actual_img = cam.read()
        if ret_val == True:
            actual_img = cv2.resize(actual_img, (int(width), int(height)))
            if (time.time() - start_time) > x:
                fps = i / (time.time() - start_time)
                start_time = time.time()
                i = 0

            flow_img = opticalflow_NN(prev_img, actual_img, args)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flow_img, "fps: "+ "{:1.2f}".format(fps), (10, 20), font, 0.5, (10, 10, 10), 2,
                       cv2.CV_AA)
            cv2.imshow('opticalflow', flow_img)
            cv2.imshow('webcam', actual_img)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            prev_img = actual_img
        else:
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel', help='path to model')
    parser.add_argument('deployproto', help='path to deploy prototxt template')
    #parser.add_argument('width', default=320, type=int)
    #parser.add_argument('height', default=240, type=int)
    parser.add_argument('--verbose', help='whether to output all caffe logging',
                        action='store_true')

    args = parser.parse_args()
    show_webcam(args, mirror=True)


if __name__ == '__main__':
    main()