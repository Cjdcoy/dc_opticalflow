import cv2
from threading import Thread
import socket
import struct
import time
import sys
import argparse
import cPickle as pickle


class VideoCamera(object):
    def __init__(self, width=320, height=240):
        self.video = cv2.VideoCapture(0)
        self.width = width
        self.height = height
        print(self.width, self.height)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            success, image = self.video.read()
            if success:
                image = cv2.resize(image, (self.width, self.height))
                return success, image


class VideoList(object):
    def __init__(self, width=320, height=240, list="videoList"):
        self.video_list = open(list, 'r').readlines()
        self.cursor = 0
        self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))
        self.width = width
        self.height = height

    def __del__(self):
        self.video.release()

    def load_new_video(self):
        self.cursor += 1
        if self.cursor < len(self.video_list):
            self.video = cv2.VideoCapture(self.video_list[self.cursor].replace("\n", ""))

    def get_frame(self):
        success, image = self.video.read()
        if success:
            image = cv2.resize(image, (self.width, self.height))
        else:
            self.load_new_video()
            success, image = self.video.read()
            if success:
                image = cv2.resize(image, (self.width, self.height))
        return success, image


class ImageList(object):
    def __init__(self, width=320, height=240, list="imageList"):
        self.image_list = open(list, 'r').readlines()
        self.cursor = 0
        self.image = cv2.imread(self.image_list[self.cursor].replace("\n", ""))
        self.width = width
        self.height = height

    def get_new_image(self):
        if self.cursor < len(self.image_list):
            self.image = cv2.imread(self.image_list[self.cursor].replace("\n", ""), 0)
            self.image = cv2.resize(self.image, (self.width, self.height))
            self.cursor += 1
            return True, self.image
        return False, self.image

    def get_frame(self):
        success, image = self.get_new_image()
        return success, image


class Streaming(Thread):
    def __init__(self):
        #Thread.__init__(self)
        if args.mode == 0:
            self.cap = VideoCamera(args.width, args.height)
        elif args.mode == 1:
            self.cap = VideoList(args.width, args.height, args.list)
        elif args.mode == 2:
            self.cap = ImageList(args.width, args.height, args.list)
        self.chrono = time.time()
        self.fps = 0

    def send_image(self, s):
        success, image = self.cap.get_frame()
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

    def fps_counter(self, args, nb_loop):
        if time.time() - self.chrono > 1:
            self.fps = nb_loop / (time.time() - self.chrono)
            self.chrono = time.time()
            if args.preview > 1:
                print(self.fps)
            return 0
        nb_loop += 1
        return nb_loop

    def preview(self, args, nb_loop, flow):
        nb_loop = self.fps_counter(args, nb_loop)
        if args.preview > 0 and args.preview != 3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flow, "fps: " + "{:1.2f}".format(self.fps), (10, 20),
                        font, 0.5, (10, 10, 10), 2,
                        cv2.LINE_AA)
        cv2.imshow('opticalflow received', flow)
        if cv2.waitKey(1) & 0xFF == 27:
            return -1
        return nb_loop

    def run(self):
        s = socket.socket()
        s.connect((args.ip, int(args.port)))
        print("Connected")
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(args.out, fourcc, 20.0, (int(self.width), int(height)))

        nb_loop = 0
        while True:
            self.send_image(s)
            flow = self.receive_image(s)
            if len(args.save):
                print(args.save)
            nb_loop = self.preview(args, nb_loop, flow)
            if nb_loop == -1:
                s.close()
                print("Socket closed, exiting.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=10000)
    parser.add_argument("-m", "--mode", help="[0] stream (default), [1] video, [2] image", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("-l", "--list", help="file containing image/video list. Format: \"path\\npath...\"", type=str, default="imageList")
    parser.add_argument("-s", "--save", help="save flow under [string].avi or save videos/images in folder [string] (empty/default: no save)", type=str, default="")
    parser.add_argument("-pre", "--preview",  help="[0] image (default), [1] image+fps, [2] print+image+fps, [3] print+image", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--width", help="width of preview / save", type=int, default=320)
    parser.add_argument("--height", help="height of preview / save", type=int, default=240)

    args = parser.parse_args()
    print(args)
    try:
        Streaming().run()
    except Exception as e:
        print(e)

