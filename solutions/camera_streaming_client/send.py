import cv2
from flask import Flask, render_template, Response
import time
from threading import Thread
import socket
import struct


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width = 320
        self.height = 160

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            success, image = self.video.read()
            if success:
                break
        #image = cv2.resize(image, (int(self.width), int(self.height)))
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


ADDRESS = ("localhost", 10000)


class Streaming(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.cap = VideoCamera()

    def run(self):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(ADDRESS)
        s.listen(1)

        print("Wait for connection")
        try:
            sc, info = s.accept()
            print("Video client connected:, info")

            while True:
                # get image surface
                image = self.cap.get_frame()
                #print(' Buffer size is %s', sc.buffer_size)
                sc.send(struct.pack('!i', len(image)))
                sc.send(image)

        except Exception as e:
            print(e)
        finally:
            # exit
            print("Closing socket and exit")
            s.close()


# --- main ---

Streaming().run()

