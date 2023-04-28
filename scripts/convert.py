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
from skl2onnx.common.data_types import FloatTensorType,Int32TensorType
from skl2onnx import convert_sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def text():
    single = True
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    # train_x,train_y = make_regression(n_samples=100,n_features=10,n_informative=5,n_targets=2,random_state=1)
    x_train = np.load(f"../data/{name}/train/feature.npy", encoding="latin1")
    y_train = np.load(f"../data/{name}/train/target.npy", encoding="latin1")
    x_test = np.load(f"../data/{name}/test/feature.npy", encoding="latin1")
    y_test = np.load(f"../data/{name}/test/target.npy", encoding="latin1")
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)

    model = joblib.load(f"../data/{name}/model/model_svr.pkl")
    sample = np.random.random(4).reshape(1, -1)
    t1 = time.perf_counter()
    sample_inference = model.predict(sample)
    t2 = time.perf_counter()

    initial_type = [('float_input', FloatTensorType([None, x_train.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type,
                          target_opset=12, verbose=False)
    with open(f"../data/{name}/model/svr.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(f"../data/{name}/model/svr.onnx", providers=["CUDAExecutionProvider"])
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
    t = np.linspace(0, 1, y_train[0].shape[0], endpoint=True)  # X轴坐标
    plt.figure(figsize=(8, 6), dpi=100)
    test_shuffle = np.random.randint(0,x_test.shape[0],5)  # 从测试集中随机抽取5个样本出来可视化结果
    for i in test_shuffle[:]:  # 看测试集上前5个的预测情况
        Dg_estimate = utils.get_line(np.arange(300, 901, step=(900 - 300) / (51 - 1)), y_predict[i], y_test[i], single)
        print(Dg_estimate)
        plt.scatter(t, y_test[i], color="green", linewidth=1, label="True")
        plt.scatter(t, y_predict_noise[i], color="black", linewidth=1, label="predict with noise")
        plt.scatter(t, y_predict[i], color="red", linewidth=1, label="predict")
        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("scattering")
        plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
        plt.show()


if __name__ == '__main__':
    text()

