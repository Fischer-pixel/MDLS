import random
import time, joblib
import numpy as np
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.datasets import make_regression
import utils


def text():
    svr0 = GridSearchCV(SVR(kernel='rbf', gamma="scale",degree=3), cv=5, n_jobs=-1,
                        param_grid={"C": [3, 4, 5]})  # np.logspace(-2, 2, 5)
    svr1 = SVR(kernel="rbf", degree=3, gamma="scale", C=4, verbose=False)
    single = True
    model = svr0 if single else svr1
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    # train_x,train_y = make_regression(n_samples=100,n_features=10,n_informative=5,n_targets=2,random_state=1)
    x_train = np.load(f"../data/{name}/train/feature.npy", encoding="latin1")
    y_train = np.load(f"../data/{name}/train/target.npy", encoding="latin1")
    x_test = np.load(f"../data/{name}/test/feature.npy", encoding="latin1")
    y_test = np.load(f"../data/{name}/test/target.npy", encoding="latin1")
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)

    # model = MultiOutputRegressor(model, n_jobs=-1)
    # model.fit(x_train, y_train)
    # print(f"the params are {model.get_params()}")
    # joblib.dump(model, f"../data/{name}/model/model_svr.pkl")

    model = joblib.load(f"../data/{name}/model/model_svr.pkl")
    sample = np.random.random(4).reshape(1, -1)
    t1 = time.perf_counter()
    sample_inference = model.predict(sample)
    t2 = time.perf_counter()
    print(f"It spends {t2 - t1} seconds to inference one sample")
    # 带上噪声污染的反演
    noise_single = np.multiply(0.01 * np.mean(x_test), np.random.normal(loc=0.0, scale=1.0, size=x_test.shape[1]))
    x_test_noise = x_test + noise_single
    y_predict_noise = model.predict(x_test_noise)
    y_predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    # 预测结果汇总输出

    V = utils.getV(y_test, y_predict)
    print(f"the test score is {score},the mean performance param is {np.mean(V)}")
    t = np.linspace(300, 900, y_train[0].shape[0], endpoint=True)  # X轴坐标
    plt.figure(figsize=(8, 6), dpi=100)
    test_shuffle = np.random.randint(0, x_test.shape[0], 5)  # 从测试集中随机抽取5个样本出来可视化结果
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
