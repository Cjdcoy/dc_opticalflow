import cv2
from threading import Thread
import socket
import struct
import time
import sys
import argparse
import cPickle as pickle


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            success, image = self.video.read()
            if success:
                image = cv2.resize(image, (int(self.width), int(self.height)))
                return image


class Streaming(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.cap = VideoCamera()
        self.chrono = time.time()
        self.fps = 0

    def send_image(self, s):
        image = self.cap.get_frame()
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
            if args.preview == 1 or args.preview == 4:
                print(self.fps)
            return 0
        nb_loop += 1
        return nb_loop

    def preview(self, args, nb_loop, flow):
        nb_loop = self.fps_counter(args, nb_loop)
        if args.preview > 2:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flow, "fps: " + "{:1.2f}".format(self.fps), (10, 20),
                        font, 0.5, (10, 10, 10), 2,
                        cv2.LINE_AA)
        if args.preview > 1:
            cv2.imshow('opticalflow received', flow)
            if cv2.waitKey(1) == 27:
                sys.exit(0)
        return nb_loop

    def run(self, ip, port):
        s = socket.socket()
        s.connect((ip, int(port)))
        print("Connected")

        nb_loop = 0
        while True:
            self.send_image(s)
            flow = self.receive_image(s)
            if args.preview > 0:
                nb_loop = self.preview(args, nb_loop, flow)
            if cv2.waitKey(1) == 27:
                sys.exit(0)
        print("Closing socket and exit")
        s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=10000)
    parser.add_argument("-pre", "--preview",  help="1 print fps, 2 render image, 3 render image+fps, 4 print fps and render image+fps", type=int, default=0, choices=[0, 1, 2, 3, 4])

    args = parser.parse_args()

    try:
        Streaming().run(args.ip, args.port)
    except Exception as e:
        print(e)

