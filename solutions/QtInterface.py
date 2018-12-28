#!/usr/bin/env python2

import sys
from PySide2.QtWidgets import QApplication, QLabel, QWidget, QLineEdit, QCheckBox, QPushButton
from PySide2.QtWidgets import QComboBox
from solutions.cloud_computed_opticalflow.optical_client import Streaming


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

        self.fps = class_widget.fps.text()

        self.ip = class_widget.ip.text()
        self.port = int(class_widget.port.text())

        self.estimation = class_widget.estimation.currentIndex()


class QtOpticalflowInterface(QWidget):
    PREVIEW = ["image",
               "image+fps",
               "print+image+fps",
               "print+image",
               ]

    MODE = ["stream",
            "video",
            "image",
            ]

    ESTIMATION = [
        "no computing estimation",
        "simple estimate",
        "complete estimation (video mode only)",
        ]

    def __init__(self):
        QWidget.__init__(self)

        self.salutation_lbl = QLabel('Welcome to Opticalflow utilise Graphic interface', self)

        self.setMinimumSize(400, 350)
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

        self.estimation_text = QLabel("estimation:", self)
        self.estimation = QComboBox(self)


        self.validation_button = QPushButton(self)

        self.set_buttons()

    def set_buttons(self):
        self.salutation_lbl.move(50, 5)  # offset the first control 5px

        self.ip_text.move(5, 35)
        self.ip.setText("localhost")
        self.ip.move(110, 35)

        self.port_text.move(5, 70)
        self.port.setText("10000")
        self.port.move(110, 70)

        self.width_text.move(5, 105)
        self.width.setText("320")
        self.width.move(110, 105)

        self.height_text.move(5, 140)
        self.height.setText("240")
        self.height.move(110, 140)

        self.save_text.move(5, 175)
        self.save.move(110, 175)
        self.save.clicked.connect(self.fps_show)

        self.name_save.move(130, 170)
        self.name_save.setText("file")
        self.name_save.hide()

        self.save_extension.move(260, 175)
        self.save_extension.hide()

        self.fps_text.move(5, 205)
        self.fps.move(110, 205)
        self.fps.setText("20")

        self.fps_text.hide()
        self.fps.hide()

        self.preview.addItems(self.PREVIEW)
        self.preview.move(110, 240)
        self.preview_text.move(5, 240)

        self.mode.addItems(self.MODE)
        self.mode.move(110, 275)
        self.mode_text.move(5, 275)

        self.estimation.addItems(self.ESTIMATION)
        self.estimation.move(110, 310)
        self.estimation_text.move(5, 310)

        self.validation_button.move(5, 340)
        self.validation_button.setText("clicked")
        self.validation_button.clicked.connect(self.reset_window)

    def get_value(self):
        return self

    def reset_window(self):
        print(self.preview)
        self.streaming = Streaming(FieldValue(self))
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

    def __delete__(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    label = QtOpticalflowInterface()
    label.show()
    sys.exit(app.exec_())
