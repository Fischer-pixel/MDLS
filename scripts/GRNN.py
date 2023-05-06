import time
import numpy as np
import math
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from pyGRNN import GRNN
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, confusion_matrix
import mplcursors
import scipy.signal as sg
from scripts import utils


class GRNN(BaseEstimator, RegressorMixin):
    def __init__(self, sigma=0.1):
        super(GRNN, self).__init__()
        self.sigma = sigma

    def distance(self, X, Y):
        '''计算两个样本之间的距离
        '''
        return np.sqrt(np.sum(np.square(X - Y), axis=0))

    def distance_mat(self, testX):
        '''计算待测试样本与所有训练样本的欧式距离
        input:trainX(mat):训练样本
              testX(mat):测试样本
        output:euclidean_distance(mat):测试样本与训练样本的距离矩阵
        两个方法中，对于大量数据的操作，sklearn中的函数明显变快,2000个样本时耗时分别为5.59s和0.006s
        '''
        # t1 = time.perf_counter()
        # m, n = np.shape(self.train_feature_)  # m: 特征集行数，也就是样本数,n:特征集列数，也就是特征数
        # p = np.shape(testX)[0]  # 测试样本集的行数，也就是测试样本数量
        # Euclidean_D = np.zeros((p, m))
        # for i in range(p):
        #     for j in range(m):
        #         Euclidean_D[i, j] = self.distance(testX[i, :], self.train_feature_[j, :])
        t2 = time.perf_counter()
        euclidean_distance = euclidean_distances(testX, self.train_feature_)
        t3 = time.perf_counter()
        return euclidean_distance

    def Gauss(self, Euclidean_D):
        '''测试样本与训练样本的距离矩阵对应的Gauss矩阵
        input:Euclidean_D(mat):测试样本与训练样本的距离矩阵
              sigma(float):Gauss函数的标准差
        output:Gauss(mat):Gauss矩阵
        两个方法中，对于大量数据的操作，sklearn中的函数明显变快,2000个样本时耗时分别为0.5s和0.007s
        '''
        # t1 = time.perf_counter()
        # m, n = np.shape(Euclidean_D)
        # Gauss = np.zeros((m, n))
        # for i in range(m):
        #     for j in range(n):
        #         Gauss[i, j] = math.exp(- Euclidean_D[i, j] / (2 * (self.sigma ** 2)))
        t2 = time.perf_counter()
        guass = np.exp(-Euclidean_D / (2 * (self.sigma ** 2)))
        t3 = time.perf_counter()
        return guass

    def sum_layer(self, Gauss):
        '''求和层矩阵，列数等于输出向量维度+1,其中0列为每个测试样本Gauss数值之和
        '''
        # m, l = np.shape(Gauss)
        # n = np.shape(self.train_target_)[1]
        # sum_mat = np.zeros((m, n + 1))
        # # 对所有模式层神经元输出进行算术求和
        # for i in range(m):
        #     sum_mat[i, 0] = np.sum(Gauss[i, :], axis=0)  # sum_mat的第0列为每个测试样本Gauss数值之和
        sum_mat0 = np.sum(Gauss, axis=1)
        # 对所有模式层神经元进行加权求和
        # for i in range(m):
        #     for j in range(n):
        #         total = 0.0
        #         for s in range(l):
        #             total += Gauss[i, s] * self.train_target_[s, j]
        #         sum_mat[i, j + 1] = total  # sum_mat的从第0列后面的列为每个测试样本Gauss加权之和
        sum_mat1 = np.matmul(Gauss, self.train_target_).T  # 加转置之后方便后续的数组行除法计算，不加时与原结果相同
        return sum_mat0, sum_mat1

    def fit(self, X=None, y=None):
        # 检查输入是否合理，允许多维度输入输出
        train_feature, train_target = check_X_y(X, y, allow_nd=True, multi_output=True)
        self.train_feature_ = train_feature
        self.train_target_ = train_target
        return self

    def predict(self, test_data):
        '''输出层输出
        input:sum_mat(mat):求和层输出矩阵
        output:output_mat(mat):输出层输出矩阵
        '''
        # check_array(test_data, allow_nd=True, ensure_min_samples=0,ensure_min_features=0)
        if len(test_data.shape) == 1:
            test_data = test_data.reshape(1, -1)
        Euclidean_D = self.distance_mat(test_data)
        Gauss = self.Gauss(Euclidean_D)
        sum_mat0, sum_mat1 = self.sum_layer(Gauss)
        # m, n = np.shape(sum_mat)
        # output_mat = np.zeros((m, n - 1))
        # for i in range(n - 1):
        #     output_mat[:, i] = sum_mat[:, i + 1] / sum_mat[:, 0]
        output_mat = (sum_mat1 / sum_mat0).T  # 鸡啊转置之后与原结果相同
        return output_mat


if __name__ == '__main__':
    single = False
    add_noise=True
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    symbol_noise = "_noise" if add_noise else ""
    sigma = 0.01423925 if single else 0.005  # 遗传算法参数寻优结果
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

    # grnn = GridSearchCV(estimator=GRNN(),
    #                     param_grid={"sigma": np.logspace(np.log10(1e-5),np.log10(0.1),20)},
    #                     scoring='neg_mean_squared_error',
    #                     cv=5, verbose=1, n_jobs=-1, refit=True
    #                     )#

    grnn = GRNN(sigma=sigma)
    grnn.fit(X=x_train, y=y_train)
    # check_is_fitted(grnn)
    print(f"the score of sigma={sigma} is {grnn.score(x_test, y_test)}")
    start = time.perf_counter()
    predict_y = grnn.predict(x_test)
    # print(predict_y.shape)
    # x_predict = np.array([[1.2578], [0.3991], [1.0852], [0.3981]])
    # predict_y = grnn.predict(x_predict)
    # t = np.linspace(100, 1200, y_test.shape[1], endpoint=True)
    # for val in range(predict_y.shape[0]):
    #     print(t[sg.argrelmax(predict_y[val])])
    # mplcursors.cursor()
    # plt.rc('text', usetex=True)  # 启用对 latex 语法的支持
    # plt.scatter(t, predict_y, color="green", linewidth=1, label=r"$predict$")
    # plt.xlabel(r"$diameter\phi (nm)$", fontsize=13)
    # plt.ylabel(r"$f(x)$", fontsize=13)
    # plt.title(r"$PSD$")
    # plt.legend(loc=1, handlelength=3, fontsize=10)  # 在右上角显示图例
    # plt.show()
    end = time.perf_counter()
    print(f"It spends {(end - start) / x_test.shape[0]} seconds to inference one sample")

    # 带上噪声污染的反演
    noise_single = np.multiply(0.01 * np.mean(x_test), np.random.normal(loc=0.0, scale=1.0, size=x_test.shape[1]))
    x_test_noise = x_test + noise_single
    y_predict_noise = grnn.predict(x_test_noise)

    # 预测结果汇总输出
    V = utils.getV(y_test, predict_y)
    print(f"the mean performance param is {np.mean(V)}")
    if single:
        t = np.linspace(100, 1200, y_test.shape[1], endpoint=True)  # X轴坐标
    else:
        t = np.linspace(300, 900, y_test.shape[1], endpoint=True)  # X轴坐标
    plt.figure(figsize=(8, 6), dpi=100)
    # np.random.seed(2)
    test_shuffle = np.random.randint(0, x_test.shape[0], 10)  # 从测试集中随机抽取5个样本出来可视化结果
    Dg_estimate_noise_list = []
    for i in test_shuffle:  # 看测试集上前5个的预测情况
        Dg_estimate, origin_mean, predict_mean = utils.get_line(t, predict_y[i], y_test[i], single,method="fit line")
        Dg_estimate_noise, _, predict_mean_noise = utils.get_line(t, y_predict_noise[i], y_test[i], single,method="fit line")
        Dg_estimate_noise_list.append(Dg_estimate_noise)
        print("%.2f" % (Dg_estimate * 100) + "%", "%.2f %.2f" % origin_mean,
              "%.2f %.2f" % predict_mean, "%.2f %.2f" % predict_mean_noise)
        plt.scatter(t, y_test[i], color="green", linewidth=1, label="True")
        plt.scatter(t, y_predict_noise[i], color="black", linewidth=1, label="predict with noise")
        plt.scatter(t, predict_y[i], color="red", linewidth=1, label="predict")
        plt.xlabel("diameter (nm)", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("user defined GRNN to predict PSD")
        plt.legend(loc=1, handlelength=3, fontsize=10)  # 在右上角显示图例
        plt.show()
    print("\n")
    for Dg_estimate_noise in Dg_estimate_noise_list:
        print("%.2f" % (Dg_estimate_noise * 100) + "%")
