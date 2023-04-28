import math,time
import scipy
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

# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']

import PyMieScatt
import warnings, sys
from scripts import utils
from tqdm import tqdm

warnings.filterwarnings("ignore")
# mie散射计算所需
sphere_index = 2.63  # 球体折射率
lambda0 = 532E-9

theta_list = [np.pi / 4]
n = len(theta_list)  # 输入向量的维度。后面会多次用到
color_map = ["red", "green", "yellow", "blue"]
label_list = ["45°", "60°", "90°", "135°"]
n_surrounding = 1.48
sphere_index = sphere_index / n_surrounding
viscosity = 146E-4
Kb = 138E-25
T = 293.2
single = True
draw_fig = True
name = "single" if single else "multiply"  # 存放数据的文件夹名字

if single:  # 单峰分布
    Dg_list = range(400, 401)
    sigma_list = [i / 100 for i in range(5, 6)]  # range只能生成整数序列
    ratio_list = [0]

data_x, data_y = [], []
for ratio in ratio_list:
    pbar = tqdm(Dg_list, total=len(Dg_list), leave=True, ncols=100, unit="个", unit_scale=False, colour="red")
    for idx, Dg in enumerate(pbar):
        pbar.set_description(f"Epoch {idx}/{len(Dg_list)}")
        start = time.perf_counter()
        for sigma in sigma_list:
            if single:
                single = utils.single_distribute(random=False, D_=Dg, sigma=sigma, start=100, end=1200, count=51)
                D_list, f_D = single.fetch()  # 获取仿真所得的粒径数据(单位是nm）和分布
            k_theta = np.zeros(n)  # 存储散射角的权重系数
            strength_list = np.zeros((n, D_list.shape[0]))  # 存储mie散射光强分数的二维数组
            h_theta = np.zeros((n, D_list.shape[0]))  # 存储电场自相关系数的二维数组
            D_theta = np.zeros(n)  # 输入GRNN神经网络中的输入向量
            G_theta = np.zeros(n)  # 存储各个散射角处光强自相关函数基线值
            for index0 in range(n):  # 开始对每个散射角进行遍历求解
                for index1 in range(D_list.shape[0]):
                    x = 2 * np.pi * D_list[index1] * n_surrounding / lambda0  # 尺寸参数：相对于波长的球体尺寸
                    s1, s2 = PyMieScatt.MieS1S2(sphere_index, x, np.cos(theta_list[index0]))
                    scatter_strength = np.power(abs(s1), 2) + np.power(abs(s2), 2)
                    strength_list[index0][index1] = scatter_strength
                light_summary = np.sum(strength_list[index0])  # 二维数组中axis=1是按行相加
                strength_list[index0] = np.divide(strength_list[index0], light_summary)  # 散射光强分数矩阵

                CIxfD = np.multiply(strength_list[index0], f_D)
                G_theta[index0] = 10E-7 * np.power(np.sum(CIxfD), 2)
                k_theta[index0] = 1 / np.sum(CIxfD)
                D_theta[index0] = np.sum(CIxfD) / \
                                  np.matmul(strength_list[index0], np.divide(f_D, D_list))  # 计算输入到GRNN网络中的特征向量数值

                h_theta[index0] = np.multiply(k_theta[index0], CIxfD)
            # print(h_theta, np.sum(h_theta, axis=1))  # 验证h_theta的累计和为1
            print(D_theta)  # 分别打印输入
            D_theta = np.multiply(1E6, D_theta)

            if draw_fig:
                t = np.logspace(np.log10(1E-5), np.log10(5), 50, base=10.0, endpoint=True)  # 时间间隔
                # plt.figure(figsize=(8, 6), dpi=100)
                buffer = []
                for i in range(n):
                    Tao = 16 * np.pi * np.power(n_surrounding, 2) * Kb * T * np.power(np.sin(theta_list[i] / 2), 2) / \
                          (3 * viscosity * lambda0 ** 2)  # 散射矢量的模
                    y = []
                    for tt in t:
                        yy = np.multiply(h_theta[i], np.exp(-Tao * tt / D_list))
                        yy = np.sum(yy)
                        y.append(yy)
                    buffer.append(y)

                #     plt.semilogx(t, np.array(y), color=color_map[i], linewidth=1, label=label_list[i])
                #
                #     plt.xlabel("delay:t(s)", fontsize=13)
                #     plt.ylabel("f(x)", fontsize=13)
                #     plt.title("dynamic light scattering:")
                #     plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
                # plt.show()

            data_x.append(buffer)
            data_y.append(f_D)
        end = time.perf_counter()
        pbar.set_postfix({"正在处理的中心粒径": Dg}, cost_time=(end - start))
data_y, data_x = np.array(data_y), np.array(data_x)
print(f"the shape of y is {data_y.shape}, the shape of x is {data_x.shape}")
print(data_y, data_x)
t = np.linspace(100, 1200, 51, endpoint=True)
plt.scatter(t, f_D, color="green", linewidth=1, label="cao")
plt.show()


def cum(x, theta):
    buffer = np.log(x)
    coef = np.polyfit(x, buffer, x.shape[0])
    poly = np.poly1d(coef)
    Tao = 16 * np.pi * np.power(n_surrounding, 2) * Kb * T * np.power(np.sin(theta / 2), 2) / \
          (3 * viscosity * lambda0 ** 2)  # 散射矢量的模
    D_caculate = -Tao / coef[-2]
    return D_caculate


x = np.load(f"../data/{name}/line_feature.npy", encoding="latin1")[0][0]
print(x.shape)
print(cum(x, np.pi / 4))

'''
[9.99870437e-01 9.99830653e-01 9.99778654e-01 9.99710691e-01
 9.99621864e-01 9.99505771e-01 9.99354047e-01 9.99155766e-01
 9.98896653e-01 9.98558071e-01 9.98115688e-01 9.97537748e-01
 9.96782835e-01 9.95796959e-01 9.94509803e-01 9.92829887e-01
 9.90638372e-01 9.87781171e-01 9.84058999e-01 9.79214966e-01
 9.72919384e-01 9.64751601e-01 9.54179076e-01 9.40534675e-01
 9.22994540e-01 9.00561225e-01 8.72060495e-01 8.36165632e-01
 7.91470250e-01 7.36638354e-01 6.70664927e-01 5.93273210e-01
 5.05441058e-01 4.09968805e-01 3.11865251e-01 2.18173396e-01
 1.36820108e-01 7.43947321e-02 3.35848999e-02 1.18978369e-02
 3.07415093e-03 5.26900939e-04 5.29971354e-05 2.67143040e-06
 5.51246054e-08 3.59989853e-10 5.38355897e-13 1.23677538e-16
 2.70073524e-21 3.20456111e-27] 0.4039796173889834'''
