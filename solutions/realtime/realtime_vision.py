from __future__ import print_function
import cv2
import os, sys, numpy as np
import argparse
from vision_module import ComputeImage
import time
from datetime import datetime, timedelta

#python2 optical_realtime.py -c ../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5 -d ../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template -pre 0 -s optical_output
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--caffemodel", help='path to model', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5")
parser.add_argument("-d", "--deployproto",  help='path to deploy prototxt template', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template")

parser.add_argument('--width', help='set width, default 320', default=320, type=int)
parser.add_argument('--height', help='set hight, default 240', default=240, type=int)
parser.add_argument("-pre", "--preview",
                    help="[-2] no preview [-1] print fps [0] image (default), [1] image+fps, [2] print+image+fps, [3] print+image",
                    type=int, default=0, choices=[-2, -1, 0, 1, 2, 3])
parser.add_argument("-s", "--save", help="save flow under [string].avi or save videos/images in folder [string] (empty/default: no save)", type=str, default="")
parser.add_argument("-f", "--fps", help="choose how many fps will have the video you receive from the server", type=int, default=20)
parser.add_argument('--verbose', help='whether to output all caffe logging', action='store_true')
args = parser.parse_args()


class FpsMetter(object):
    def __init__(self):
        self.chrono = time.time()
        self.fps = 0

    def get_fps(self, nb_loop):
        if time.time() - self.chrono > 1:
            self.fps = nb_loop / (time.time() - self.chrono)
            self.chrono = time.time()
            if args.preview > 1 or args.preview == -1:
                print(self.fps)
            return 0
        nb_loop += 1
        return nb_loop


class OpticalRealtime(object):
    def __init__(self, args_m):
        realtime = __import__("vision_module", globals(), locals(), ['ComputeImage'], -1)
        reload(realtime)
        self.args = args_m
        self.computeImage = realtime.ComputeImage()  # COMPUTE IMAGE IS THE MODULE YOU HAVE TO LOAD IN ORDER TO SELECT YOUR ALOORITHM
        self.cap = cv2.VideoCapture(0)
        self.fpsMetter = FpsMetter()
        self.width = self.args.width
        self.height = self.args.height
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.videoFps = self.args.fps
        if len(self.args.save) > 0:
            self.out = cv2.VideoWriter(self.args.save + ".avi", self.fourcc, args.fps, (args.width, args.height))

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        while True:
            success, image = self.cap.read()
            if success:
                image = cv2.resize(image, (self.width, self.height))
                return success, image

    def save_flow(self, flow):
        self.out.write(flow)

    def preview(self, nb_loop, flow):
        nb_loop = self.fpsMetter.get_fps(nb_loop)
        if self.args.preview > 0 and self.args.preview != 3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flow, "fps: " + "{:1.2f}".format(self.fpsMetter.fps), (10, 20),
                        font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        if self.args.preview > -1:
            cv2.imshow('opticalflow received', flow)
            if cv2.waitKey(1) & 0xFF == 27:
                return -1
        return nb_loop

    def run(self):
        success, prev_img = self.get_frame()
        if not success:
            sys.exit(0)
        nb_loop = 0
        while True:
            ret_val, actual_img = self.get_frame()
            if ret_val:
                actual_img = cv2.resize(actual_img, (int(self.args.width), int(self.args.height)))
                flow_img = self.computeImage.run(prev_img, actual_img, self.args)
                nb_loop = self.preview(nb_loop, flow_img)
                if len(self.args.save) > 0:
                    self.save_flow(flow_img)
                if nb_loop == -1:
                    break
                prev_img = actual_img
            else:
                break
        cv2.destroyAllWindows()


def main():
    OpticalRealtime(args).run()


if __name__ == '__main__':
    main()