from __future__ import print_function
import cv2
import os, sys, numpy as np
import argparse
from vision_module import ComputeImage
import time
from datetime import datetime, timedelta
from tqdm import tqdm, trange

#python2 optical_video.py -c ../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5 -d ../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template -pre 0 -s test
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--caffemodel", help='path to model', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5")
parser.add_argument("-d", "--deployproto",  help='path to deploy prototxt template', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template")

parser.add_argument('--width', help='set width, default 320', default=320, type=int)
parser.add_argument('--height', help='set hight, default 240', default=240, type=int)
parser.add_argument("-pre", "--preview",
                    help="[-2] no preview [-1] print fps [0] image (default), [1] image+fps, [2] print+image+fps, [3] print+image",
                    type=int, default=0, choices=[-2, -1, 0, 1, 2, 3])
parser.add_argument("-l", "--list", help="file containing image/video list. Format: \"path\\npath...\"", type=str, default="video_list_example")
parser.add_argument("-s", "--save", help="save flow under [string].avi or save videos/images in folder [string] (empty/default: no save)", type=str, default="")
parser.add_argument("-f", "--fps", help="choose how many fps will have the video you receive from the server", type=int, default=20)
parser.add_argument("-e", "--estimation", help="[0] no computing estimation [1] simple estimate [2] complete estimation", type=int, default=0, choices=[0, 1, 2])

parser.add_argument('--verbose', help='whether to output all caffe logging',
                    action='store_true')

args = parser.parse_args()


class FpsMetter(object):
    def __init__(self, args_m):
        self.args = args_m
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
            if self.args.preview > 1 or self.args.preview == -1:
                print("fps: {:1.2f}".format(self.fps))
            return 0
        nb_loop += 1
        return nb_loop


class VideoList(object):
    def __init__(self, args_m):
        realtime = __import__("vision_module", globals(), locals(), ['ComputeImage'], 0)
        reload(realtime)
        self.args = args_m
        self.computeImage = ComputeImage() # COMPUTE IMAGE IS THE MODULE YOU HAVE TO LOAD IN ORDER TO SELECT YOUR ALOORITHM
        self.fpsMetter = FpsMetter(self.args)
        self.video_list = open(self.args.list, 'r').readlines()
        self.cursor = 0
        self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
        self.width = self.args.width
        self.height = self.args.height
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.videoFps = self.args.fps
        self.estimation = False
        #pbar
        self.pbar_isset = False

        if self.args.estimation > 0:
            self.estimation = True
        if len(self.args.save) > 0:
            if not os.path.exists(self.args.save):
                os.makedirs(self.args.save)
            self.out = cv2.VideoWriter(self.args.save + "/" + str(self.cursor) + ".avi", self.fourcc, int(self.args.fps), (int(self.args.width), int(self.args.height)))

    def load_new_video(self, save):
        self.cursor += 1
        if self.cursor < len(self.video_list):
            self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
            if save:
                self.out.release()
                self.out = cv2.VideoWriter(self.args.save + "/" + str(self.cursor) + ".avi", self.fourcc, int(self.args.fps), (int(self.args.width), int(self.args.height)))

    def get_frame(self, save):
        if self.args.estimation > 0 and self.pbar_isset:
            self.pbar.update()
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
            if self.args.estimation == 2:
                print("video " + str(i) + ": " + str(
                    frames) + " frames (" + "{:1.2f}".format(
                    frames / fps) + " seconds)")
            total_nb_frame += frames
        print("\nThere are " + str(total_nb_frame) + " frames to compute")
        print("Estimated compute time (day, hour, min, sec):")
        self.pbar = tqdm(desc='compute estimation', total=total_nb_frame, unit='f')
        self.pbar_isset = True
        sec = timedelta(seconds=total_nb_frame / fps)
        d = datetime(1, 1, 1) + sec
        print("{:02d}".format(d.day - 1) + ":" + "{:02d}".format(
            d.hour) + ":" + "{:02d}".format(d.minute) + ":" + "{:02d}".format(
            d.second))

    def preview(self, nb_loop, flow):
        nb_loop = self.fpsMetter.get_fps(nb_loop)
        if self.fpsMetter.init_finished and self.estimation: #does not estimate if the solution takes more than 1 second per image
            self.estimate_compute_time(self.fpsMetter.fps)
        if self.args.preview > 0 and self.args.preview != 3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flow, "fps: " + "{:1.2f}".format(self.fpsMetter.fps), (10, 20),
                        font, 0.5, (255, 0, 255), 2)
        if self.args.preview > -1:
            cv2.imshow('image received', flow)
            if cv2.waitKey(1) & 0xFF == 27:
                return -1
        return nb_loop

    def run(self):
        save = False
        if len(self.args.save) > 0:
            save = True
        ret_val, prev_img = self.get_frame(save)
        nb_loop = 0
        while True:
            ret_val, actual_img = self.get_frame(save)
            if ret_val:
                flow_img = self.computeImage.run(prev_img, actual_img, self.args)
                if len(self.args.save) > 0:
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
    VideoList(args).run()


if __name__ == '__main__':
    main()
