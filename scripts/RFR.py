import time, joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import utils
import mplcursors
import scipy.signal as sg


def text():
    single = True
    material = "water"
    angle = "single"
    model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    # train_x,train_y = make_regression(n_samples=100,n_features=10,n_informative=5,n_targets=2,random_state=1)
    x_train = np.load(f"../data/water/multi-angle/{name}/dataset/train/feature.npy", encoding="latin1")
    y_train = np.load(f"../data/water/multi-angle/{name}/dataset/train/target.npy", encoding="latin1")
    x_test = np.load(f"../data/water/multi-angle/{name}/dataset/test/feature.npy", encoding="latin1")
    y_test = np.load(f"../data/water/multi-angle/{name}/dataset/test/target.npy", encoding="latin1")
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)

    # model.fit(x_train, y_train)
    # print(f"the params are {model.get_params()}")
    # joblib.dump(model, f"../data/water/multi-angle/{name}/model/rfr.pkl")

    model = joblib.load(f"../data/water/multi-angle/{name}/model/rfr.pkl")
    sample = np.random.random(4).reshape(1, -1)
    t1 = time.perf_counter()
    _ = model.predict(sample)
    t2 = time.perf_counter()
    print(f"It spends {t2 - t1} seconds to inference one sample")
    # 带上噪声污染的反演
    noise_single = np.multiply(0.01 * np.mean(x_test), np.random.normal(loc=0.0, scale=1.0, size=x_test.shape[1]))
    x_test_noise = x_test + noise_single
    y_predict = model.predict(x_test)
    y_predict_noise = model.predict(x_test_noise)
    # x_predict = np.array([[0.866]])
    # predict_y = model.predict(x_predict)
    # print(predict_y)
    # t = np.linspace(100, 1200, y_test.shape[1], endpoint=True)
    # print(t[sg.argrelmax(predict_y[0])])
    # mplcursors.cursor()
    # plt.rc('text', usetex=True)  # 启用对 latex 语法的支持
    # plt.scatter(t, predict_y, color="green", linewidth=1, label=r"$predict$")
    # plt.xlabel(r"$diameter\phi (nm)$", fontsize=13)
    # plt.ylabel(r"$f(x)$", fontsize=13)
    # plt.title(r"$PSD$")
    # plt.legend(loc=1, handlelength=3, fontsize=10)  # 在右上角显示图例
    # plt.show()
    score = model.score(x_test, y_test)
    # 预测结果汇总输出

    V = utils.getV(y_test, y_predict)
    print(f"the test score is {score},the mean performance param is {np.mean(V)}")
    if single:
        t = np.linspace(100, 1200, y_test.shape[1], endpoint=True)  # X轴坐标
    else:
        t = np.linspace(300, 900, y_test.shape[1], endpoint=True)  # X轴坐标
    plt.figure(figsize=(8, 6), dpi=100)
    plt.rc('text', usetex=False)  # 启用对 latex 语法的支持
    test_shuffle = np.random.randint(0, x_test.shape[0], 10)  # 从测试集中随机抽取5个样本出来可视化结果
    Dg_estimate_noise_list = []
    for i in test_shuffle:  # 看测试集上前5个的预测情况
        Dg_estimate, origin_mean, predict_mean = utils.get_line(t, y_predict[i], y_test[i], single)
        Dg_estimate_noise, _, predict_mean_noise = utils.get_line(t, y_predict_noise[i], y_test[i], single)
        Dg_estimate_noise_list.append(Dg_estimate_noise)
        print("%.2f" % (Dg_estimate * 100) + "%", "%.2f" % origin_mean, "%.2f" % predict_mean, "%.2f" % predict_mean_noise)
        plt.scatter(t, y_test[i], color="green", linewidth=1, label="True")
        plt.scatter(t, y_predict_noise[i], color="black", linewidth=1, label="predict with noise")
        plt.scatter(t, y_predict[i], color="red", linewidth=1, label="predict")
        plt.xlabel("diameter (nm)", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("Dynamic light scattering curve")
        plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
        plt.show()
    print("\n")
    for Dg_estimate_noise in Dg_estimate_noise_list:
        print("%.2f" % (Dg_estimate_noise * 100) + "%")
    # 模型可视化
    # from six import StringIO
    # from sklearn.tree import export_graphviz
    # import pydotplus
    # import os
    #
    # # 执行一次
    # os.environ['PATH'] = os.pathsep + r"D:\program files\Graphviz\bin/"
    # model = RandomForestRegressor()
    # pipe = Pipeline([('regressor', model)])  # ,('scaler', StandardScaler()), ('reduce_dim', PCA())
    # pipe.fit(x_train, y_train)
    # dot_data = StringIO()
    # export_graphviz(pipe.named_steps['regressor'].estimators_[0],
    #                 out_file=dot_data)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('tree.png')


if __name__ == '__main__':
    text()
