import cv2
import os
import numpy as np

#Computeimage is a generic class that allows us easily change the algorithm within the testing environment
#you can load the properties you want placing a file .csv
#csv format : type,value. Type can either be INT or STR
#class variables will be initiated from the csv in their declaration order

class ComputeImage(object):
    def __init__(self, values_to_load='compute_image.csv'):
        self.blurX, self.blurY = 1, 1
        self.dilatation = 2
        self.first_run = True
        if os.path.isfile(values_to_load):
            self.init_class_from_csv(values_to_load)

    def init_class_from_csv(self, values_to_load):
        content = open(values_to_load, 'r').read().split(',')
        it = 0

        for dir in self.__dir__():
            if dir in self.__dict__ and it < len(content):
                if content[it] == 'int':
                    self.__dict__[dir] = int(content[it + 1])
                if content[it] == 'str':
                    self.__dict__[dir] = str(content[it + 1])
                it += 2

    def run(self, prev_frame, actual_frame, args=None):
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, actual_frame, None, 0.5, 3, 3, 6, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr