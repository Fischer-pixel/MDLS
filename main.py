import logging
import os
import glob
import pandas as pd
import time
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import miepython
import PyMieScatt
import warnings
import sys
import matplotlib
import scripts.utils
from scripts import utils
from tqdm import tqdm
from collections import namedtuple, OrderedDict, Counter
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sko import GA, PSO
import mplcursors
import cv2
import scipy.signal as sg
from scripts.GRNN import GRNN
from scripts import utils
from mutli_length import generator
import readfile
from mainwin import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
# matplotlib.use("Qt5Agg")
plt.rc('text', usetex=True)  # 启用对 latex 语法的支持


def generate_dataset():
    t1 = time.perf_counter()
    data = generator(material="oil", theta_list=[np.pi / 4, np.pi / 3, np.pi / 2, 3 * np.pi / 4],
                     single=True, add_noise=True, draw_fig=False, method="cumulate")
    # data_x, data_y = data.generate()
    num_angle = "one-angle" if len(data.theta_list) <= 1 else "multi-angle"
    symbol_single = "single" if data.single else "multiply"
    symbol_noise = "_noise" if data.add_noise else ""
    symbol_line = "line_" if data.method == "line" else ""
    experiment,draw = False,True
    # np.save(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/{symbol_line}feature.npy", data_x)
    # np.save(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/{symbol_line}target.npy", data_y)

    data_x = np.load(
        f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/{symbol_line}feature.npy")
    data_y = np.load(
        f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/{symbol_line}target.npy")
    if data.add_noise:
        x_train, y_train, x_test, y_test = [], [], [], []
        if data.method == "cumulate":
            for index0 in range(data_x.shape[0]):
                D_theta, f_D = data_x[index0], data_y[index0]
                x_train.append(D_theta)
                y_train.append(f_D)
                for index in range(int(data.configs["noise_sample_num"]) + 1):
                    D_theta_copy = D_theta  # 作为独立计算的备份
                    noise_single = np.multiply(float(data.configs["noise_level"]) * np.mean(D_theta),
                                               np.random.normal(loc=0.0, scale=1.0, size=data.n))
                    D_theta_copy = D_theta_copy + noise_single
                    if index == int(data.configs["noise_sample_num"]):
                        x_test.append(D_theta_copy)
                        y_test.append(f_D)
                    else:
                        x_train.append(D_theta_copy)
                        y_train.append(f_D)
        else:  # data.method == "line"
            pass
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    else:
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.80, random_state=1)

    x_train = np.load(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/train/{symbol_line}feature{symbol_noise}.npy")
    y_train = np.load(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/train/{symbol_line}target{symbol_noise}.npy")
    x_test = np.load(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/test/{symbol_line}feature{symbol_noise}.npy")
    y_test = np.load(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/test/{symbol_line}target{symbol_noise}.npy")

    # np.save(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/train/{symbol_line}feature{symbol_noise}.npy",
    #         x_train)
    # np.save(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/train/{symbol_line}target{symbol_noise}.npy",
    #         y_train)
    # np.save(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/test/{symbol_line}feature{symbol_noise}.npy",
    #         x_test)
    # np.save(f"./data/{data.material}/{num_angle}/{symbol_single}/dataset/test/{symbol_line}target{symbol_noise}.npy",
    #         y_test)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    t2 = time.perf_counter()
    print(f"It spends {(t2 - t1):.3f} seconds to generate dataset")
    t = np.linspace(100, 1200, y_test.shape[1], endpoint=True)
    sk_model = "GRNN"
    if sk_model == "GRNN":
        sigma = 0.01423925 if data.single else 0.005  # 遗传算法参数寻优结果
        grnn = GRNN(sigma=sigma)
        grnn.fit(X=x_train, y=y_train)
        print(f"the score of sigma={sigma} is {grnn.score(x_test, y_test)*100:.2f}%")
        start = time.perf_counter()
        y_predict = grnn.predict(x_test)
        end = time.perf_counter()
        print(f"It spends {(end - start) / x_test.shape[0]} seconds to inference one sample")
        V = scripts.utils.getV(y_test, y_predict)
        print(f"the mean performance param is {np.mean(V) * 100:.2f}%")
    else:
        # model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        # model.fit(x_train, y_train)
        # print(f"the params are {model.get_params()}")
        # joblib.dump(model,
        #             f"./data/{data.material}/{num_angle}/{symbol_single}/model/{symbol_line}rfr{symbol_noise}.pkl")
        model = joblib.load(
            f"./data/{data.material}/{num_angle}/{symbol_single}/model/{symbol_line}rfr{symbol_noise}.pkl")
        score = model.score(x_test, y_test)
        # 预测结果汇总输出
        start = time.perf_counter()
        y_predict = model.predict(x_test)
        end = time.perf_counter()
        print(f"It spends {(end - start) / x_test.shape[0]} seconds to inference one sample")
        V = utils.getV(y_test, y_predict)
        print(f"the test score is {score * 100:.2f}%,the mean performance param is {np.mean(V) * 100:.2f}%")

    performance_li = np.zeros(y_predict.shape[0])
    for index in range(y_predict.shape[0]):
        performance, _, _ = scripts.utils.get_line(t, y_predict[index], y_test[index], data.single,
                                                   method="use maximum")
        performance_li[index] = performance
    print(f"测试集峰均值相对误差为{np.mean(performance_li) * 100:.2f}%")

    if draw:
        test_shuffle = np.random.randint(0, x_test.shape[0], 5)  # 从测试集中随机抽取5个样本出来可视化结果
        min_index = np.argmin(V)
        max_index = np.argmax(V)
        print("%.2f"%V[min_index], "%.2f"%V[max_index])
        min_max_index = np.array([min_index, max_index])
        for i in test_shuffle:  # 看测试集上前5个的预测情况
            plt.figure(figsize=(8, 6), dpi=100)
            Dg_estimate, origin_mean, predict_mean = utils.get_line(t, y_predict[i], y_test[i], data.single)
            print("%.2f" % (Dg_estimate * 100) + "%", "%.2f" % origin_mean, "%.2f" % predict_mean)
            plt.scatter(t, y_test[i], color="green", linewidth=1, label="True PSD")
            plt.scatter(t, y_predict[i], color="red", linewidth=1, label="Predict PSD")
            plt.xlabel("Diameter (nm)", fontsize=13)
            plt.ylabel("f(x)", fontsize=13)
            plt.title("Use GRNN to Predict PSD")
            plt.legend(loc=1, handlelength=3, fontsize=10)  # 在右上角显示图例
            plt.show()

    if experiment:
        stamp, acf = readfile.read_fit()
        predict_x = utils.cumulate1(acf, data.Tao[0], x=stamp)
        predict_y = grnn.predict(predict_x)
        print(predict_x, data.Tao[0], predict_y.shape)
        print(t[sg.argrelmax(predict_y)])


class MainWindow(QMainWindow,Ui_MainWindow):
    send_start_meg = pyqtSignal

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowIcon(QIcon("logo.ico"))
        self.setupUi(self)

        self.start.clicked.connect(self.change_status_start)
        # self.comboBox_key.currentIndexChanged.connect(self.change_key)
        # self.doubleSpinBox_threshold.valueChanged.connect(self.change_threshold)

    def change_status_start(self):  # 按下了启动按钮
        self.start.setEnabled(False)
        self.run()
        QApplication.processEvents()

    def change_status_stop(self):

        self.start_bn.setEnabled(True)
        self.stop_bn.setEnabled(False)

    def change_cutsize(self):
        b = self.comboBox_cutsize.currentText().split(",")

    def change_px(self):
        self.P_x = self.doubleSpinBox_px.value()

    def change_delay(self):
        self.delay = self.spinBox_delay.value()

    def run(self):
        frame = cv2.imread("result.png")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[2] == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888

        result = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, qformat)
        result = QPixmap(result).scaled(self.psd.width(), self.psd.height())
        # self.label.setPixmap(QPixmap.fromImage(result))
        self.psd.setPixmap(result)
        self.output_text.append("466.33"+"nm")  # 展示中心粒径的反演结果
#       # self.output_text.clear()


class Thread(QThread):
    def __init__(self):
        super(Thread, self).__init__()

    def run(self):
        MainWindow().run()


def main():
    # generate_dataset()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setWindowTitle("变压器油粒径反演软件")
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
