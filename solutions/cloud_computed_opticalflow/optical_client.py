import cv2
from threading import Thread
import socket
import struct
import time
import sys
import argparse
import os
import cPickle as pickle
from datetime import datetime, timedelta


class VideoCamera(object):
    def __init__(self, width=320, height=240, save="", fps=20):
        self.video = cv2.VideoCapture(0)
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = fps
        if len(save) > 0:
            self.out = cv2.VideoWriter(save + ".avi", self.fourcc, fps, (width, height))

    def __del__(self):
        self.video.release()

    def get_frame(self, save=None):
        while True:
            success, image = self.video.read()
            if success:
                image = cv2.resize(image, (self.width, self.height))
                return success, image

    def save_flow(self, flow):
        self.out.write(flow)


class VideoList(object):
    def __init__(self, width=320, height=240, list="videoList", save="", fps=20):
        self.video_list = open(list, 'r').readlines()
        self.cursor = 0
        self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = fps
        self.save = save
        if len(save) > 0:
            if not os.path.exists(save):
                os.makedirs(save)
            self.out = cv2.VideoWriter(self.save + "/" + str(self.cursor) + ".avi", self.fourcc, fps, (width, height))

    def __del__(self):
        self.video.release()

    def load_new_video(self, save):
        self.cursor += 1
        if self.cursor < len(self.video_list):
            self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
            if save:
                self.out.release()
                self.out = cv2.VideoWriter(self.save + "/" + str(self.cursor) + ".avi", self.fourcc, self.fps, (self.width,  self.height))

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

    def save_flow(self, flow):
        self.out.write(flow)


class ImageList(object):
    def __init__(self, width=320, height=240, list="imageList", save=""):
        self.image_list = open(list, 'r').readlines()
        self.cursor = 0
        self.image = cv2.imread(self.image_list[self.cursor].replace("\n", ""))
        self.width = width
        self.height = height
        self.save = save
        if len(save) > 0:
            if not os.path.exists(save):
                os.makedirs(save)

    def get_new_image(self):
        if self.cursor < len(self.image_list):
            self.image = cv2.imread(self.image_list[self.cursor].replace("\n", ""), 0)
            self.image = cv2.resize(self.image, (self.width, self.height))
            self.cursor += 1
            return True, self.image
        return False, self.image

    def get_frame(self, save=None):
        success, image = self.get_new_image()
        return success, image

    def save_flow(self, flow):
        print(self.save + "/" + str(self.cursor) + ".png")
        cv2.imwrite(self.save + "/" + str(self.cursor) + ".png", flow, [cv2.IMWRITE_PNG_COMPRESSION, 9])


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


class Streaming(Thread):
    def __init__(self):
        self.fpsMetter = FpsMetter()
        self.estimation = False
#for algorithms that need to load a neural network we pass the first loop otherwise the
# computing estimation could be rigged
        self.first_loop = True
        if args.estimation > 0:
            self.estimation = True
        #Thread.__init__(self)
        if args.mode == 0:
            self.cap = VideoCamera(args.width, args.height, args.save, args.fps)
        elif args.mode == 1:
            self.cap = VideoList(args.width, args.height, args.list, args.save, args.fps)
        elif args.mode == 2:
            self.cap = ImageList(args.width, args.height, args.list, args.save)
        self.chrono = time.time()
        self.fps = 0

    def send_image(self, s):
        success, image = self.cap.get_frame(args.save)
        if not success:
            s.close()
            sys.exit(0)
        serialized_data = pickle.dumps(image, protocol=2)
        s.send(struct.pack('!i', len(serialized_data)))
        s.send(serialized_data)

    def receive_image(self, s):
        len_str = s.recv(4)
        size = struct.unpack('!i', len_str)[0]
        blob = b''
        while size > 0:
            if size >= 4096:
                data = s.recv(4096)
            else:
                data = s.recv(size)
            if not data:
                break
            size -= len(data)
            blob += data

        if sys.version_info.major < 3:
            unserialized_blob = pickle.loads(blob)
        else:
            unserialized_blob = pickle.loads(blob, encoding='bytes')
        return unserialized_blob

    def fps_counter(self, nb_loop):
        if time.time() - self.chrono > 1:
            self.fps = nb_loop / (time.time() - self.chrono)
            self.chrono = time.time()
            if args.preview > 1:
                print(self.fps)
            return 0
        nb_loop += 1
        return nb_loop

    def estimate_compute_time(self, fps):
        self.estimation = False
        total_nb_frame = 0
        print("Calculating compute time...\nEstimated FPS: " + "{:1.2f}".format(fps) + "\n")
        for i in range(0, len(self.cap.video_list)):
            cap = cv2.VideoCapture(self.cap.video_list[i].replace("\n", ""))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.estimation == 2:
                print("video " + str(i) + ": " + str(frames) + " frames (" + "{:1.2f}".format(frames / fps) + " seconds)")
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
        # does not estimate for the first loop, and this is reserved to video computing
        if self.fpsMetter.init_finished and self.estimation and args.mode == 1:
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

    def run(self):
        s = socket.socket()
        s.connect((args.ip, int(args.port)))
        nb_loop = 0

        print("Connected")
        while True:
            self.send_image(s)
            flow = self.receive_image(s)
            if len(args.save) > 0:
                self.cap.save_flow(flow)
            nb_loop = self.preview(nb_loop, flow)
            if nb_loop == -1:
                break
            self.first_loop = False
        s.close()
        cv2.destroyAllWindows()
        print("Socket closed, windows destroyed, exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=10000)
    parser.add_argument("--width", help="width of preview / save", type=int, default=320)
    parser.add_argument("--height", help="height of preview / save", type=int, default=240)
    parser.add_argument("-pre", "--preview", help="[-1] no preview [0] image (default), [1] image+fps, [2] print+image+fps, [3] print+image", type=int, default=0, choices=[-1, 0, 1, 2, 3])

    parser.add_argument("-m", "--mode", help="[0] stream (default), [1] video, [2] image", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("-l", "--list", help="file containing image/video list. Format: \"path\\npath...\"", type=str, default="video_list_example")

    parser.add_argument("-s", "--save", help="save flow under [string].avi or save videos/images in folder [string] (empty/default: no save)", type=str, default="")
    parser.add_argument("-f", "--fps", help="choose how many fps will have the video you receive from the server", type=int, default=20)

    parser.add_argument("-e", "--estimation", help="[0] no computing estimation [1] simple estimate [2] complete estimation (video mode only)", type=int, default=0, choices=[0, 1, 2])

    args = parser.parse_args()
    print(args)
    try:
        Streaming().run()
    except Exception as e:
        print(e)

