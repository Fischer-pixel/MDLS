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
    svr0 = GridSearchCV(SVR(kernel='rbf', gamma="scale", degree=3), cv=5, n_jobs=-1,
                        param_grid={"C": [3, 4, 5]})  # np.logspace(-2, 2, 5)
    svr1 = SVR(kernel="rbf", degree=3, gamma="scale", C=4, verbose=False)
    single = True
    add_noise = False
    model = svr1
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
    # data_x, data_y = make_regression(n_samples=5, n_features=4, n_informative=5, n_targets=2, random_state=1)
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.80, random_state=1)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model = MultiOutputRegressor(model, n_jobs=-1)
    model.fit(x_train, y_train)
    print(f"the params are {model.get_params()}")
    joblib.dump(model, f"../data/oil/multi-angle/{name}/model/svr{symbol_noise}.pkl")

    model = joblib.load(f"../data/oil/multi-angle/{name}/model/svr{symbol_noise}.pkl")
    sample = np.random.random(4).reshape(1, -1)
    t1 = time.perf_counter()
    sample_inference = model.predict(sample)
    t2 = time.perf_counter()
    print(f"It spends {t2 - t1} seconds to inference one sample")
    # 带上噪声污染的反演
    # noise_single = np.multiply(0.01 * np.mean(x_test), np.random.normal(loc=0.0, scale=1.0, size=x_test.shape[1]))
    # x_test_noise = x_test + noise_single
    # y_predict_noise = model.predict(x_test_noise)

    y_predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    # 预测结果汇总输出

    V = utils.getV(y_test, y_predict)
    print(f"the test score is {score},the mean performance param is {np.mean(V)}")
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
        plt.title("use Tikhonov to predict PSD")
        plt.legend(loc=1, handlelength=3, fontsize=10)  # 在右上角显示图例
        plt.show()
    print("\n")
    # for Dg_estimate_noise in Dg_estimate_noise_list:
    #     print("%.2f" % (Dg_estimate_noise * 100) + "%")


if __name__ == '__main__':
    text()
