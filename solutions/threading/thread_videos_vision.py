from __future__ import print_function
from tqdm import tqdm
import cv2
import os, sys, numpy as np
import argparse
from vision_module import ComputeImage
import tensorflow as tf
import time
import requests
from datetime import datetime, timedelta
from threading import Thread
from uuid import uuid4

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
parser.add_argument("-s", "--save", help="save flow under [string].mov or save videos/images in folder [string] (empty/default: no save)", type=str, default="")
parser.add_argument("-f", "--fps", help="choose how many fps will have the video you receive from the server", type=int, default=20)
parser.add_argument("-e", "--estimation", help="[0] no computing estimation [1] simple estimate [2] complete estimation", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("-t", "--thread", help="number of threads", type=int, default=1)

parser.add_argument('--verbose', help='whether to output all caffe logging',
                    action='store_true')

args = parser.parse_args()


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
            if args.preview > 1 or args.preview == -1:
                print("fps: {:1.2f}".format(self.fps))
            return 0
        nb_loop += 1
        return nb_loop


    def add_image(self, flow_image, width=320, height=240, fps=20):
        if self.fall_detected:
            self.image_to_add_post_fall -= 1
            if self.image_to_add_post_fall < 0:
                self.image_to_add_post_fall = 0
                self.fall_detected = False
                self.write_send_fall_video(width, height, fps)
        if self.list_len < self.nb_rush_image:
            self.image_list.append(flow_image)
            self.list_len += 1
        else:
            self.image_list.pop(0)
            self.image_list.append(flow_image)

    def toggle_fall(self):
        if not self.fall_detected:
            print("fall detected")
            self.fall_detected = True
            self.image_to_add_post_fall = self.nb_rush_image_after_fall
        else:
            print("fall already detected")

class VideoList(Thread):
    def __init__(self, video_list):
        super().__init__()
        self.is_ready = False
        self.flow_result = None
        self.keep_running = True
        self.detectorImages = []
        self.fall_probability = 0.
        self.prev_img = None
        self.computeImage = ComputeImage() # COMPUTE IMAGE IS THE MODULE YOU HAVE TO LOAD IN ORDER TO SELECT YOUR ALOORITHM
        self.fpsMetter = FpsMetter()
        self.video_list = video_list
        self.cursor = 0
        #self.list = os.listdir('/home/blind/ai_visualsolutions/dataset/Coffee_room_01/Videos/')
        self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
        #self.video = cv2.VideoCapture('/home/blind/ai_visualsolutions/dataset/Coffee_room_01/Videos/' + self.list[self.cursor].replace("\n", ""))
        self.width = args.width
        self.height = args.height
        self.fourcc = cv2.VideoWriter_fourcc(*'FLV1')
        self.videoFps = args.fps
        self.pbar = None
        self.estimation = False
        self.estimated = False
        if args.estimation > 0:
            self.estimation = True
        if len(args.save) > 0:
            if not os.path.exists(args.save):
                os.makedirs(args.save)
            self.out = cv2.VideoWriter(args.save + "/" + str(self.cursor) + ".mov", self.fourcc, args.fps, (args.width, args.height), True)

    def load_new_video(self, save):
        self.cursor += 1
        if self.cursor < len(self.video_list):
            self.video.release()
            self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
            #self.video = cv2.VideoCapture('/home/blind/ai_visualsolutions/dataset/Coffee_room_01/Videos/' + self.list[self.cursor])
            print(self.video_list[self.cursor])
            if save:
                self.out.release()
                self.out = cv2.VideoWriter(args.save + "/" + str(self.cursor) + ".mov", self.fourcc, self.videoFps, (self.width,  self.height))

    def get_frame(self, save):
        success, image = self.video.read()
        if self.estimated:
            self.estimate_progress_bar()
        if success:
            image = cv2.resize(image, (self.width, self.height))
        else:
            self.load_new_video(save)
            success, image = self.video.read()
            if success:
                image = cv2.resize(image, (self.width, self.height))
                self.prev_img = image
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

    def estimate_progress_bar(self):
        self.pbar.update(1)

    def estimate_compute_time(self, fps):
        self.estimation = False
        total_nb_frame = 0
        print("Calculating compute time...\nEstimated FPS: " + "{:1.2f}".format(
            fps) + "\n")
        for i in range(0, len(self.video_list)):
            cap = cv2.VideoCapture(self.video_list[i].replace("\n", ""))
            #cap = cv2.VideoCapture('/home/blind/ai_visualsolutions/dataset/Coffee_room_01/Videos/'+self.list[i].replace("\n", ""))
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
        # those two lines are only for the progress bar
        self.pbar = tqdm(total=total_nb_frame, unit='frame')
        self.estimated = True

    def preview(self, nb_loop, flow):
        #add flow to the fall video

        font = cv2.FONT_HERSHEY_SIMPLEX
        nb_loop = self.fpsMetter.get_fps(nb_loop)
        if self.fpsMetter.init_finished and self.estimation: #does not estimate if the solution takes more than 1 second per image
            self.estimate_compute_time(self.fpsMetter.fps)
        if args.preview > 0 and args.preview != 3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flow, "fps: " + "{:1.2f}".format(self.fpsMetter.fps), (10, 40),
                        font, 0.5, (0, 204, 0), 2)
        if len(args.save) > 0:
            self.save_flow(flow)  # save video
        self.flow_result = cv2.resize(flow, (720, 480))
        self.is_ready = True
        return nb_loop

    def run(self):
        print('__ run __')
        save = False
        if len(args.save) > 0:
            save = True
        ret_val, self.prev_img = self.get_frame(save)
        nb_loop = 0
        while self.keep_running:
            ret_val, actual_img = self.get_frame(save)
            if ret_val:
                flow_img = self.computeImage.run(self.prev_img, actual_img, args)
                nb_loop = self.preview(nb_loop, flow_img)   #preview / estimated / save
                if nb_loop == -1:
                    break

                self.prev_img = actual_img
            else:
                break
        self.video.release()
        cv2.destroyAllWindows()


def main():
    try:
        video_list = open(args.list, 'r').readlines()
        thread_list = []
        blocksize = len(video_list) // args.thread

        for i in range(args.thread):
            target_list = video_list[blocksize * i:(blocksize * i) + blocksize]
            print(list)
            videoList = VideoList(target_list)
            videoList.setName(str(i))
            thread_list.append(videoList)

        for thread in thread_list:
            thread.start()

        exit = False
        while not exit:
            for thread in thread_list:
                if thread.is_ready:
                    if args.preview > -1:
                        cv2.imshow(thread.getName(), thread.flow_result)
                        if cv2.waitKey(1) & 0xFF == 27:
                            exit = True

        for thread in thread_list:
            thread.keep_running = False
            thread.join()


    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
