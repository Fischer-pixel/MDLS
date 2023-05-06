import skl2onnx
import onnx
import sklearn
import random
import time, joblib
import numpy as np
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.datasets import make_regression
import utils
from sklearn.linear_model import LogisticRegression
import numpy
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType,Int32TensorType,DoubleTensorType
from skl2onnx import convert_sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def text():
    single = False
    add_noise = True
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    symbol_noise = "_noise" if add_noise else ""
    x_train = np.load(
        f"../data/oil/multi-angle/{name}/dataset/train/feature{symbol_noise}.npy")
    y_train = np.load(
        f"../data/oil/multi-angle/{name}/dataset/train/target{symbol_noise}.npy")
    x_test = np.load(
        f"../data/oil/multi-angle/{name}/dataset/test/feature{symbol_noise}.npy")
    y_test = np.load(
        f"../data/oil/multi-angle/{name}/dataset/test/target{symbol_noise}.npy")
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model = joblib.load(f"../data/oil/multi-angle/{name}/model/svr{symbol_noise}.pkl")
    sample = np.random.random(4).reshape(1, -1)
    t1 = time.perf_counter()
    sample_inference = model.predict(sample)
    t2 = time.perf_counter()

    initial_type = [('float_input', FloatTensorType([None, x_train.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type,
                          target_opset=12, verbose=False)
    with open(f"../data/oil/multi-angle/{name}/model/svr{symbol_noise}.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(f"../data/oil/multi-angle/{name}/model/svr{symbol_noise}.onnx",
                               providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    t3 = time.perf_counter()
    y_predict = sess.run([label_name], {input_name: x_test.astype(numpy.float32)})[0]
    t4 = time.perf_counter()
    print(f"It spends {t2 - t1} seconds by CPU and {(t4-t3)/x_test.shape[0]} seconds by CUDA to inference one sample")
    # 带上噪声污染的反演
    noise_single = np.multiply(0.01 * np.mean(x_test), np.random.normal(loc=0.0, scale=1.0, size=x_test.shape[1]))
    x_test_noise = x_test + noise_single
    y_predict_noise = sess.run([label_name], {input_name: x_test_noise.astype(numpy.float32)})[0]
    # 预测结果汇总输出

    V = utils.getV(y_test, y_predict)
    print(f"the mean performance param is {np.mean(V)}")
    if single:
        t = np.linspace(100, 1200, y_test.shape[1], endpoint=True)  # X轴坐标
    else:
        t = np.linspace(300, 900, y_test.shape[1], endpoint=True)  # X轴坐标

    test_shuffle = np.random.randint(0, x_test.shape[0], 10)  # 从测试集中随机抽取5个样本出来可视化结果
    # Dg_estimate_noise_list = []
    for i in test_shuffle:  # 看测试集上前5个的预测情况
        Dg_estimate, origin_mean, predict_mean = utils.get_line(t, y_predict[i], y_test[i], single, method="fit line")
        # Dg_estimate_noise, _, predict_mean_noise = utils.get_line(t, y_predict_noise[i], y_test[i], single,
        #                                                           method="fit line")
        # Dg_estimate_noise_list.append(Dg_estimate_noise)
        if single:
            print("%.2f" % (Dg_estimate * 100) + "%", "%.2f" % origin_mean, "%.2f" % predict_mean)
        else:
            print("%.2f" % (Dg_estimate * 100) + "%", "%.2f %.2f" % origin_mean, "%.2f %.2f" % predict_mean)
        plt.scatter(t, y_test[i], color="green", linewidth=1, label="True")
        # plt.scatter(t, y_predict_noise[i], color="black", linewidth=1, label="predict with noise")
        plt.scatter(t, y_predict[i], color="red", linewidth=1, label="predict")
        plt.xlabel("diameter (nm)", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("user defined GRNN to predict PSD")
        plt.legend(loc=1, handlelength=3, fontsize=10)  # 在右上角显示图例
        plt.show()
    print("\n")
    # for Dg_estimate_noise in Dg_estimate_noise_list:
    #     print("%.2f" % (Dg_estimate_noise * 100) + "%")

if __name__ == '__main__':
    text()

