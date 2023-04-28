import math
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.misc import derivative
from scipy.optimize import fminbound
import pandas as pd
import statsmodels.tsa.api as smt
import scipy.signal as sg
import readfile

# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']


class single_distribute:
    def __init__(self, D_=590, sigma=0.15, start=300, end=900, count=51, random=False):
        self.D_ = 1E-9 * D_
        self.D = np.linspace(1E-9 * start, 1E-9 * end, 100, endpoint=True)
        self.sigma = sigma
        self.start, self.end, self.count = start, end, count
        self.random = random

    def calculate(self, D, D_, sigma):
        y0 = np.exp(-(np.power(np.log(D / D_), 2) / (2 * np.power(sigma, 2)))) / \
             (np.sqrt(2 * np.pi) * sigma * D * 10E5)
        return y0

    def fig(self):
        plt.figure(figsize=(8, 6), dpi=100)
        y = self.calculate(self.D, self.D_, self.sigma)
        plt.plot(self.D, y, color="gray", linewidth=1.0, label="single")  # marker="^",dashes=[2,1]
        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("unimodal distribution")
        plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
        # plt.savefig("result.jpg")
        plt.show()

    def fetch(self):
        if self.random:
            seed_x = np.random.randint(self.start, self.end, size=self.count)
        else:
            seed_x = np.arange(self.start, self.end + 1, step=(self.end - self.start) / (self.count - 1))
        seed_x = np.multiply(seed_x, 1E-9)
        y = self.calculate(seed_x, self.D_, self.sigma)
        # plt.scatter(seed_x, y, color="red", linewidths=1)
        # plt.xlabel("diameter (m)", fontsize=13)
        # plt.ylabel("f(x)", fontsize=13)
        # plt.title("single peak psd")
        # plt.show()
        return seed_x, y


class multi_distribute:
    def __init__(self, d_list=[500, 800], sigma_list=[0.13, 0.045],
                 ratio_list=[0.8, 0.2], start=300, end=900, count=51, random=False):
        self.D_ = [1E-9 * i for i in d_list]
        self.D = np.linspace(30E-8, 100E-8, 200, endpoint=True)
        self.sigma = sigma_list
        self.ratio = ratio_list
        self.start, self.end, self.count = start, end, count
        self.random = random

    def calculate(self, x):
        def cal(D, D_, sigma):
            y0 = np.exp(-(np.power(np.log(D / D_), 2) / (2 * np.power(sigma, 2)))) / \
                 (np.sqrt(2 * np.pi) * sigma * D * 10E5)
            return y0

        num = 0
        for i in range(2):
            num = num + np.multiply(np.array(self.ratio[i]), cal(x, self.D_[i], self.sigma[i]))
        return num

    def fig(self):
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(self.D, self.calculate(self.D), color="red", linewidth=1.0,
                 label="multiply")  # marker="^",dashes=[2,1]
        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("multimodal distribution")
        plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
        plt.show()

    def fetch(self):
        if self.random:
            seed_x = np.random.randint(self.start, self.end, size=self.count)
        else:
            seed_x = np.arange(self.start, self.end + 1, step=(self.end - self.start) / (self.count - 1))
        seed_x = np.multiply(seed_x, 1E-9)
        y = self.calculate(seed_x)
        # plt.scatter(seed_x, y, color="red", linewidths=1)
        # plt.xlabel("diameter (m)", fontsize=13)
        # plt.ylabel("f(x)", fontsize=13)
        # plt.title("double peak psd")
        # plt.show()
        return seed_x, y


def distribute(single=True, draw=False):
    # 需要令lognorm中的参数s=sigma,loc=0,scale=exp(mu)
    from scipy import stats
    # standard deviation of normal distribution
    if not single:
        sigma0 = 0.13
        sigma1 = 0.045
        # mean of normal distribution
        mu0 = 0.5
        mu1 = 0.8
        portion0, portion1 = 80, 20
        print(np.exp(mu0))
        # hopefully, total is the value where you need the cdf

        r0 = stats.lognorm.rvs(s=sigma0, loc=0, scale=np.exp(mu0), size=portion0)
        r1 = stats.lognorm.rvs(s=sigma1, loc=0, scale=np.exp(mu1), size=portion1)
        result = np.append(np.log(r0), np.log(r1))
        print(result)
        mean, loc, mu = stats.lognorm.fit(r0, floc=0)
        print(mean, loc, np.log(mu))
        if draw:  # 是否绘制概率密度函数
            x = np.linspace(0, 5, 1000, endpoint=True)
            y0 = stats.lognorm.pdf(x, s=sigma0, loc=0, scale=np.exp(mu0))
            y1 = stats.lognorm.pdf(x, s=sigma1, loc=0, scale=np.exp(mu1))
            plt.plot(x, y0, color="red", linewidth=1.0, label=str(mu0) + "um")  # marker="^",dashes=[2,1]
            plt.plot(x, y1, color="green", linewidth=1.0, label=str(mu1) + "um")  # marker="^",dashes=[2,1]
            plt.show()
        return result
        # samples = np.random.lognormal(mean=mu, sigma=sigma, size=50)
    else:
        sigma = 0.15
        mu = 0.59
        print(np.exp(mu))
        # hopefully, total is the value where you need the cdf
        r = stats.lognorm.rvs(s=sigma, loc=0, scale=np.exp(mu), size=100)
        result = np.log(r)
        print(result)
        mean, loc, mu = stats.lognorm.fit(r, floc=0)
        print(mean, loc, np.log(mu))
        return result


# 获取性能参数
def getV(f_D, f_estimate_D):
    if len(f_D.shape) == 1:
        f_D = f_D.reshape(1, -1)
        f_estimate_D = f_estimate_D.reshape(1, -1)
    fenzi = np.sum(np.power((f_D - f_estimate_D), 2), axis=1)  # 当输入二维数组时按行相加
    fenmu = np.sum(np.power(f_D, 2), axis=1)
    V = np.sqrt(np.true_divide(fenzi, fenmu))
    return V


def fmaxbound(func, x1, x2, args=(), **kwargs):
    return fminbound(lambda x: -func(x, *args), x1, x2, **kwargs)  # 对函数加个负号相当于求极大值


def get_line(D_i, f_estimate_D, f_D, single, method="use maximum"):  # 获取拟合曲线以及均值粒径
    # 获取样条函数，k：样条函数的平滑度 1<=k<=5
    f_predict = InterpolatedUnivariateSpline(D_i, f_estimate_D, k=3)
    f_origin = InterpolatedUnivariateSpline(D_i, f_D, k=3)
    x = np.linspace(D_i.min(), D_i.max(), D_i.shape[0], endpoint=True)
    # plt.plot(x, f_predict(x), color="red", linewidth=2.0, label="fit line")
    # plt.plot(D_i, f_D, color="green", linewidth=2.0, label="true psd")
    # plt.legend()
    # plt.show()

    if single:
        Dg_estimate = fmaxbound(f_predict, D_i.min(), D_i.max(), xtol=1)  # 单峰分布就直接返回最大值点
        Dg_origin = fmaxbound(f_origin, D_i.min(), D_i.max(), xtol=1)
        performance = abs(Dg_origin - Dg_estimate) / Dg_origin
        return performance, Dg_origin, Dg_estimate
    else:
        if method == "fit line":
            Dg_estimate_min = fminbound(f_predict, D_i.min(), D_i.max(), xtol=1)  # 反演函数的最小值点，此为求该函数在区间上的最小值点,用以区间分割
            Dg_origin_min = fminbound(f_origin, D_i.min(), D_i.max(), xtol=1)  # 原函数最小值点
            _ = x[sg.argrelmin(f_origin(x))[0]]
            predict0 = fmaxbound(f_predict, D_i.min(), Dg_estimate_min, xtol=1)  # ,disp=False,xtol=1e-5
            predict1 = fmaxbound(f_predict, Dg_estimate_min, D_i.max(), xtol=1)
            origin0 = fmaxbound(f_origin, D_i.min(), Dg_origin_min, xtol=1)
            origin1 = fmaxbound(f_origin, Dg_origin_min, D_i.max(), xtol=1)

        else:
            origin0, origin1 = x[sg.argrelmax(f_origin(x))[0]]
            predict0, predict1 = x[sg.argrelmax(f_predict(x))[0]]
        performance = (abs(origin0 - predict0) / origin0 + abs(origin1 - predict1) / origin1) / 2
        return performance, (origin0, origin1), (predict0, predict1)


def cumulate1(line_x, tao, x=None):
    if x is None:
        x = np.logspace(np.log10(1E-5), np.log10(5), 50, base=10.0, endpoint=True)

    buffer = np.log(line_x)
    coef = np.polyfit(x, buffer, line_x.shape[0])
    poly = np.poly1d(coef)
    # print(coef,poly)
    D_caculate = -tao / coef[-2]
    return np.array([D_caculate])


def generate_brown():
    loc, scale = 0, 1
    # plt.rc("text", usetex=True)

    x = np.logspace(np.log10(1E-5), np.log10(5), 50, base=10.0, endpoint=True)
    y = (1 / (np.sqrt(2 * np.pi) * scale)) * np.exp(-np.square(x) / (2 * np.square(scale)))
    acf = smt.stattools.acf(y, nlags=y.shape[0] - 1, fft=True)  # 求自相关函数，即使序列均值不是0也可以正常运行
    y1 = np.load(f"../data/single/line_feature.npy", encoding="latin1")[0][0]
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(1):
        li_x = np.random.normal(loc=0, scale=1, size=50)
        acf_noise = smt.stattools.acf(li_x, nlags=li_x.shape[0] - 1, fft=True)
        plt.semilogx(x, li_x, color="red", linewidth=1.0, label="original noise")
        y = (y1 + acf_noise) / 2
        plt.semilogx(x, acf_noise, color="yellow", linewidth=1.0, label="acf noise")
        plt.semilogx(x, y1, color="black", linewidth=1.0, label="acf curve")
        plt.semilogx(x, y, color="green", linewidth=1.0, label="result")
        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("The auto correlation curve is solved for the standard normal distribution")
        plt.legend()  # 显示图例
        plt.show()

    for i in range(10):
        plt.figure(figsize=(10, 6), dpi=100)
        li_x = np.random.normal(loc=0, scale=1, size=50)
        plt.semilogx(x, y1, color="yellow", linewidth=1.0, label="true curve")
        plt.semilogx(x, li_x, color="green", linewidth=1.0, label="noise curve")
        plt.axhline(0, linewidth=1)
        plt.axhline(1, linewidth=1)
        result = li_x * 0.1 + y1
        plt.semilogx(x, result, color="red", linewidth=1.0, label="acf curve with noise")
        import warnings

        warnings.filterwarnings("ignore")
        cao1 = cumulate1(abs(result), np.pi / 4)
        buffer = np.argwhere(result < 0)
        result[buffer] = 0
        cao2 = cumulate1(result, np.pi / 4)
        # if cao >= 0:
        #     print(result.max(), result.min())
        print(cao1,
              np.load(f"../data/single/feature_one_angle.npy", encoding="latin1")[0][0])
        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("The auto correlation curve is solved for the standard normal distribution")
        plt.legend()  # 显示图例
        plt.show()


def cumulate():
    sphere_index = 2.63  # 球体折射率
    lambda0 = 532E-9
    num_angle = 13
    # 多角度多峰分布
    # theta_list = np.pi * np.array([i / 180 for i in range(20, 141, 10)])
    theta_list = [np.pi / 4, np.pi / 3, np.pi / 2, 3 * np.pi / 4]  # 以弧度计算
    Dg_list = [400, 500, 600, 700]
    n = len(theta_list)  # 输入向量的维度。后面会多次用到
    color_map = ["red", "green", "yellow", "blue"]
    label_list = ["45°", "60°", "90°", "135°"]
    n_surrounding = 1.48
    sphere_index = sphere_index / n_surrounding
    viscosity = 20E-3
    Kb = 138E-25
    T = 298
    single = True
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    # np.random.seed(1)  # 确定随机种子，确保实验可重复
    # train_x,train_y = make_regression(n_samples=100,n_features=10,n_informative=5,n_targets=2,random_state=1)
    # data_x = np.load(f"../data/{name}/line_feature.npy", encoding="latin1")
    # data_y = np.load(f"../data/{name}/feature_one_angle.npy", encoding="latin1")  # 平均粒径
    stamp, acf = readfile.read_fit()
    buffer = np.log(acf)
    coef = np.polyfit(stamp, buffer, stamp.shape[-1])
    poly = np.poly1d(coef)
    # print(coef,poly)
    Tao = 16 * np.pi * np.power(n_surrounding, 2) * Kb * T * np.power(np.sin(theta_list[0] / 2), 2) / \
          (3 * viscosity * lambda0 ** 2)  # 散射矢量的模
    D_caculate = -Tao / coef[-2]
    print(Tao,D_caculate, "\n")  # 源程序copy到另一个脚本运行结果就不对了，暂未查出原因
    # x = np.logspace(np.log10(1E-5), np.log10(5), 50, base=10.0, endpoint=True)
    # test_shuffle = np.random.randint(0, data_x.shape[0], 10)  # 从测试集中随机抽取5个样本出来可视化结果
    # for index in test_shuffle:
    #     for i in range(1):
    #         buffer = np.log(data_x[index][i])
    #         coef = np.polyfit(x, buffer, data_x.shape[-1])
    #         poly = np.poly1d(coef)
    #         # print(coef,poly)
    #         Tao = 16 * np.pi * np.power(n_surrounding, 2) * Kb * T * np.power(np.sin(theta_list[i] / 2), 2) / \
    #               (3 * viscosity * lambda0 ** 2)  # 散射矢量的模
    #         D_caculate = -Tao / coef[-2]
    #         D = data_y[index][i]
    #         print(D_caculate, D, "\n")  # 源程序copy到另一个脚本运行结果就不对了，暂未查出原因


def cursor():
    import mplcursors

    data = np.outer(range(10), range(1, 5))

    fig, ax = plt.subplots()
    lines = ax.plot(data)
    ax.set_title("Click somewhere on a line.\nRight-click to deselect.\n"
                 "Annotations can be dragged.")

    mplcursors.cursor(lines)  # or just mplcursors.cursor()

    plt.show()


if __name__ == '__main__':
    # single1 = single_distribute()
    # single1.fetch()
    # multi = multi_distribute()
    # multi.fetch()
    # result = distribute()
    cumulate()
    # generate_brown()
    # cursor()
    #
