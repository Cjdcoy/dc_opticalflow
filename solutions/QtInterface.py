#!/usr/bin/env python2

import sys
import glob, os

from PySide2.QtWidgets import QApplication, QLabel, QWidget, QLineEdit, QCheckBox, QPushButton
from PySide2.QtWidgets import QComboBox
from cloud_computed_vision.optical_client import Streaming
from realtime.realtime_vision import OpticalRealtime
from shutil import copyfile


class FieldValue(object):
    def __init__(self, class_widget):
        self.preview = class_widget.preview.currentIndex()
        self.mode = class_widget.mode.currentIndex()

        self.width = class_widget.width.text()
        self.height = class_widget.height.text()

        if not class_widget.save.isChecked():
            self.save = ""
        else:
            self.save = class_widget.name_save.text() + ".avi"

        if class_widget.mode.currentText() == "video" or class_widget.mode.currentText() == "realtime":
            self.list = class_widget.video_file.text()

        self.fps = class_widget.fps.text()

        self.ip = class_widget.ip.text()
        self.port = int(class_widget.port.text())

        self.estimation = class_widget.estimation.currentIndex()


class QtOpticalflowInterfaceCloud(QWidget):
    PREVIEW = ["image",
               "image+fps",
               "print+image+fps",
               "print+image",
               ]

    MODE = ["stream",
            "video",
            ]

    ESTIMATION = [
        "no computing estimation",
        "simple estimate",
        "complete estimation (video mode only)",
        ]

    os.chdir("../modules")
    VISION_MODULES = glob.glob("*.py")
    os.chdir("../solutions")

    def __init__(self):
        QWidget.__init__(self)

        self.salutation_lbl = QLabel('Welcome to Opticalflow utilise Graphic interface', self)

        self.compute_location_text = QLabel("compute location:", self)
        self.compute_location = QComboBox(self)

        self.setMinimumSize(400, 400)
        self.setWindowTitle('GUI opticalflow utilities')

        self.ip_text = QLabel("ip:", self)
        self.ip = QLineEdit(self)

        self.port_text = QLabel("port:", self)
        self.port = QLineEdit(self)

        self.width_text = QLabel("width:", self)
        self.width = QLineEdit(self)

        self.height_text = QLabel("height:", self)
        self.height = QLineEdit(self)

        self.save_text = QLabel("save:", self)
        self.save = QCheckBox(self)

        self.save_extension = QLabel(".avi", self)

        self.name_save = QLineEdit(self)

        self.fps_text = QLabel("fps:", self)
        self.fps = QLineEdit(self)

        self.preview_text = QLabel("preview:", self)
        self.preview = QComboBox(self)

        self.mode_text = QLabel("mode:", self)
        self.mode = QComboBox(self)

        self.video_file = QLineEdit(self)

        self.estimation_text = QLabel("estimation:", self)
        self.estimation = QComboBox(self)

        self.module_vision_text = QLabel("vision module:", self)
        self.module_vision = QComboBox(self)

        self.validation_button = QPushButton(self)

        self.set_buttons()

    def set_buttons(self):
        self.salutation_lbl.move(50, 5)  # offset the first control 5px

        self.compute_location_text.move(5, 35)
        self.compute_location.move(120, 30)
        self.compute_location.addItems(["local", "cloud"])
        self.compute_location.currentIndexChanged.connect(self.cloud_ip)

        self.ip_text.move(5, 70)
        self.ip.setText("localhost")
        self.ip.move(110, 70)
        self.ip.hide()
        self.ip_text.hide()

        self.port_text.move(5, 105)
        self.port.setText("10000")
        self.port.move(110, 105)
        self.port.hide()
        self.port_text.hide()

        self.width_text.move(5, 140)
        self.width.setText("320")
        self.width.move(110, 140)

        self.height_text.move(5, 175)
        self.height.setText("240")
        self.height.move(110, 175)

        self.save_text.move(5, 205)
        self.save.move(110, 205)
        self.save.clicked.connect(self.fps_show)

        self.name_save.move(130, 200)
        self.name_save.setText("file")
        self.name_save.hide()

        self.save_extension.move(260, 205)
        self.save_extension.hide()

        self.fps_text.move(5, 240)
        self.fps.move(110, 240)
        self.fps.setText("20")

        self.fps_text.hide()
        self.fps.hide()

        self.preview.addItems(self.PREVIEW)
        self.preview.move(110, 275)
        self.preview_text.move(5, 275)

        self.mode.addItems(self.MODE)
        self.mode.move(110, 310)
        self.mode_text.move(5, 310)
        self.mode.currentIndexChanged.connect(self.video_mode)

        self.video_file.move(190, 310)
        self.video_file.setText("videos_to_computed_videos/video_list_example")
        self.video_file.hide()

        self.estimation.addItems(self.ESTIMATION)
        self.estimation.move(110, 345)
        self.estimation_text.move(5, 345)

        self.module_vision.addItems(self.VISION_MODULES)
        self.module_vision.move(110, 375)
        self.module_vision_text.move(5, 375)

        self.validation_button.move(5, 400)
        self.validation_button.setText("clicked")
        self.validation_button.clicked.connect(self.reset_window)

    def get_value(self):
        return self

    def reset_window(self):
        copyfile("../modules/" + self.module_vision.currentText(), "./realtime/vision_module.py")
        if self.compute_location.currentText() == "cloud":
            self.streaming = Streaming(FieldValue(self))
        else:
            self.streaming = OpticalRealtime(FieldValue(self))

        self.streaming.run()

    def fps_show(self):
        if self.save.isChecked():
            self.fps_text.show()
            self.fps.show()
            self.name_save.show()
            self.save_extension.show()
        else:
            self.fps_text.hide()
            self.fps.hide()
            self.name_save.hide()
            self.save_extension.hide()

    def cloud_ip(self):
        if self.compute_location.currentText() == "local":
            self.ip.hide()
            self.ip_text.hide()
            self.port.hide()
            self.port_text.hide()
        else:
            self.ip.show()
            self.ip_text.show()
            self.port.show()
            self.port_text.show()

    def video_mode(self):
        if self.mode.currentText() == "video":
            self.video_file.show()
        else:
            self.video_file.hide()

    def __delete__(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    label = QtOpticalflowInterfaceCloud()
    label.show()
    sys.exit(app.exec_())
