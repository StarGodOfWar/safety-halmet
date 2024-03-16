import argparse
import platform
import shutil
import time
from numpy import random
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
from pathlib import Path
import numpy
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np
import time
#模型函数
def load_model(
        weights=ROOT / 'best.pt',  # model.pt path(s)权重路径
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path数据集配置文件路径
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference 使用 FP16 半精度推断
        dnn=False,  # use OpenCV DNN for ONNX inference 使用 OpenCV DNN 进行 ONNX 推断

):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    return model, stride, names, pt, jit, onnx, engine


def run(model, img, stride, pt,
        imgsz=(640, 640),  # inference size (height, width) 推断尺寸（高、宽）
        conf_thres=0.65,  # confidence threshold 置信度阈值
        iou_thres=0.15,  # NMS IOU threshold NMS IOU 阈值
        max_det=1000,  # maximum detections per image 每张图像的最大检测数
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu 
        classes=None,  # filter by class: --class 0, or --class 0 2 3 按类别筛选
        agnostic_nms=False,  # class-agnostic NMS 类别无关 NMS
        augment=False,  # augmented inference #增强推断
        half=False,  # use FP16 half-precision inference 
        ):

    cal_detect = []

    device = select_device(device)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Set Dataloader数据加载器
    im = letterbox(img, imgsz, stride, pt)[0]

    # Convert转换图像
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=augment)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process detections 检测结果
    for i, det in enumerate(pred):  # detections per image 每张图像的检测结果
        if len(det):
            # Rescale boxes from img_size to im0 size 缩放尺寸
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

            # Write results

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]}'
                lbl = names[int(cls)]
                #print(lbl)
                #if lbl not in [' Chef clothes',' clothes',' chef hat',' hat']:
                    #continue
                cal_detect.append([label, xyxy,float(conf)])
    return cal_detect

#目标检测
def det_yolov7(info1):
    global model, stride, names, pt, jit, onnx, engine
    if info1[-3:] in ['jpg','png','jpeg','tif','bmp']:
        image = cv2.imread(info1)  # 读取识别对象
        try:
            results = run(model, image, stride, pt)  # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
            for i in results:
                box = i[1]
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                color = [255,0,0]
                if i[0] == 'helmet':
                    color = [0, 0, 255]
                    i[0] = 'NO helmet'
                    ui.printf('警告！检测到工人未戴安全帽')
                if i[0] == 'head':
                    color = [0, 255, 0]
                    i[0] = 'Helmet'
                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        except:
            pass
        ui.showimg(image)
    if info1[-3:] in ['mp4','avi']:
        capture = cv2.VideoCapture(info1)
        while True:
            _, image = capture.read()
            if image is None:
                break
            try:
                results = run(model, image, stride, pt)  # 识别，返回多个数组每个第一个为结果，第二个为坐标位置
                for i in results:
                    box = i[1]
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    color = [255, 0, 0]
                    if i[0] == 'helmet':
                        color = [0, 0, 255]
                        i[0] = 'NO helmet'
                        ui.printf('警告！检测到工人未戴安全帽')
                    if i[0] == 'head':
                        color = [0, 255, 0]
                        i[0] = 'Helmet'
                    cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                    cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            except:
                pass
            ui.showimg(image)
            QApplication.processEvents()

#继承自QThread的线程类，用于执行图像检测
class Thread_1(QThread):  # 线程1
    def __init__(self,info1):
        super().__init__()
        self.info1=info1
        self.run2(self.info1)

    def run2(self, info1):
        result = []
        result = det_yolov7(info1)

#UI界面
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/carui.png\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(50, 30, 840, 450))
        self.label_1.setStyleSheet("background:rgba(0,0,0,0);")
        self.label_1.setFrameShape(QtWidgets.QFrame.Box)
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1.setObjectName("label_1")


        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 480, 840, 450))
        self.label_2.setStyleSheet("background:rgba(0,0,0,0);")
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")


        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(910, 480, 350, 450))
        self.textBrowser.setStyleSheet("background:rgba(0,0,0,0);")
        self.textBrowser.setObjectName("textBrowser")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(910, 30, 150, 40))
        self.pushButton.setStyleSheet("background:rgba(0,0,0,0.1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1110, 30, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(0,0,0,0.1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(910, 100, 150, 40))
        self.pushButton_3.setStyleSheet("background:rgba(0,0,0,0.1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1110, 100, 150, 40))
        self.pushButton_4.setStyleSheet("background:rgba(0,0,0,0.1);border-radius:10px;padding:2px 4px;")
        self.pushButton_4.setObjectName("pushButton_4")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(910, 150, 350, 300))
        self.label_3.setStyleSheet("background:rgba(0,0,0,0);")
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "安全帽佩戴检测系统"))

        self.pushButton.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_2.setText(_translate("MainWindow", "视频检测"))
        self.pushButton_3.setText(_translate("MainWindow", "实时检测"))
        self.pushButton_4.setText(_translate("MainWindow", "重置/选择对象"))

        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.click_1)
        self.pushButton_2.clicked.connect(self.click_3)
        self.pushButton_3.clicked.connect(self.click_2)
        self.pushButton_4.clicked.connect(self.openfile)

    def click_1(self):
        global model, stride, names, pt, jit, onnx, engine
        start_time = time.time()
        image = cv2.imread(filepath)  # 读取识别对象
        ui.showimg1(image)
        dai_count = 0
        no_count = 0
        try:
            results = run(model, image, stride, pt)  # 识别，返回多个数组每个第一个为结果，第二个为坐标位置
            for i in results:
                box = i[1]
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                color = [255, 0, 0]
                if i[0] == 'helmet':
                    color = [0, 0, 255]
                    i[0] = 'NO helmet'
                    no_count += 1
                    ui.printf('警告！检测到工人未戴安全帽')
                if i[0] == 'head':
                    color = [0, 255, 0]
                    dai_count += 1
                    i[0] = 'Helmet'
                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        except:
            pass
        ui.showimg2(image)
        tabel = cv2.imread('./tem/2.png')

        end_time = time.time()
        t = end_time - start_time
        cv2.putText(tabel, str(t)[:4] + 's' , (260, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(tabel, str(dai_count), (280, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(tabel, str(no_count), (280, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        ui.showimg3(tabel)
        QApplication.processEvents()

    def click_2(self):
        capture = cv2.VideoCapture(0)
        start_time = time.time()
        while True:
            _, image = capture.read()
            dai_count = 0
            no_count = 0
            if image is None:
                break

            ui.showimg1(image)
            image = cv2.resize(image,(640,640))
            results = run(model, image, stride, pt)  # 识别，返回多个数组每个第一个为结果，第二个为坐标位置
            for i in results:
                box = i[1]
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                color = [255, 0, 0]
                if i[0] == 'helmet':
                    color = [0, 0, 255]
                    i[0] = 'NO helmet'
                    no_count += 1
                    ui.printf('警告！检测到工人未戴安全帽')
                if i[0] == 'head':
                    color = [0, 255, 0]
                    dai_count += 1
                    i[0] = 'Helmet'
                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            ui.showimg2(image)
            tabel = cv2.imread('./tem/2.png')

            cv2.putText(tabel, str(dai_count), (280, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(tabel, str(no_count), (280, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ui.showimg3(tabel)
            QApplication.processEvents()

        end_time = time.time()
        t = end_time - start_time
        cv2.putText(tabel, str(t)[:4] + 's', (260, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        ui.showimg3(tabel)
        QApplication.processEvents()

    def click_3(self):

        capture = cv2.VideoCapture(filepath)
        start_time = time.time()
        while True:
            _, image = capture.read()
            dai_count = 0
            no_count = 0
            if image is None:
                break
            ui.showimg1(image)

            results = run(model, image, stride, pt)  # 识别，返回多个数组每个第一个为结果，第二个为坐标位置
            for i in results:
                box = i[1]
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                color = [255, 0, 0]
                if i[0] == 'helmet':
                    color = [0, 0, 255]
                    i[0] = 'NO helmet'
                    no_count += 1
                    ui.printf('警告！检测到工人未戴安全帽')
                if i[0] == 'head':
                    color = [0, 255, 0]
                    dai_count += 1
                    i[0] = 'Helmet'
                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            ui.showimg2(image)
            tabel = cv2.imread('./tem/2.png')

            cv2.putText(tabel, str(dai_count), (280, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(tabel, str(no_count), (280, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ui.showimg3(tabel)
            QApplication.processEvents()

        end_time = time.time()
        t = end_time - start_time
        cv2.putText(tabel, str(t)[:4] + 's', (260, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        ui.showimg3(tabel)
        QApplication.processEvents()

    def openfile(self):
        global sname, filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        show = cv2.imread('./tem/1.png')
        tabel = cv2.imread('./tem/2.png')
        ui.showimg1(show)
        ui.showimg2(show)
        ui.showimg3(tabel)
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("重置成功！当前选择的文件路径是：%s" % filepath)



    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg1(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 800
        else:
            ratio = n_height / 800
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_1.setPixmap(QPixmap.fromImage(new_img))

    def showimg2(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 800
        else:
            ratio = n_height / 800
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_2.setPixmap(QPixmap.fromImage(new_img))

    def showimg3(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        new_img = _image.scaled(n_width, n_height, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QPixmap.fromImage(new_img))

    def click(self):
        global filepath
        try:
            self.thread_1.quit()
        except:
            pass
        self.thread_1 = Thread_1(filepath)  # 创建线程
        self.thread_1.wait()
        self.thread_1.start()  # 开始线程


if __name__ == "__main__":
    global model, stride, names, pt, jit, onnx, engine
    model, stride, names, pt, jit, onnx, engine = load_model()  # 加载模型
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
