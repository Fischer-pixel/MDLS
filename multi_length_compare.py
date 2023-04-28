import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import miepython
import warnings, sys
from scripts import utils
import PyMieScatt

warnings.filterwarnings("ignore")
# mie散射计算所需
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

# t = np.linspace(0, 3, 10000, endpoint=True)  # 时间间隔
# plt.figure(figsize=(8, 6), dpi=100)
# for index, Dg in enumerate(Dg_list):
#     single = utils.single_distribute(D_=Dg, sigma=0.15, start=300, end=900, count=51, random=False)
#     D_list, f_D = single.fetch()  # 获取仿真所得的粒径数据(单位是nm）和分布
#     k_theta = np.zeros(n)  # 存储散射角的权重系数
#     strength_list = np.zeros((n, D_list.shape[0]))  # 存储mie散射光强分数的二维数组
#     h_theta = np.zeros((n, D_list.shape[0]))  # 存储电场自相关系数的二维数组
#
#     for index1 in range(D_list.shape[0]):
#         x = 2 * np.pi * D_list[index1] * n_surrounding / lambda0  # 尺寸参数：相对于波长的球体尺寸
#         # s1, s2 = miepython.mie_S1_S2(sphere_index, x, np.cos(theta_list[index0]))
#         s1, s2 = PyMieScatt.MieS1S2(sphere_index, x, np.cos(theta_list[0]))
#         scatter_strength = np.power(abs(s1), 2) + np.power(abs(s2), 2)
#         strength_list[0][index1] = scatter_strength
#
#     light_summary = np.sum(strength_list[0])  # 二维数组中axis=1是按行相加,为每个散射角处散射光强分数的总和
#     strength_list[0] = np.divide(strength_list[0], light_summary)  # 散射光强分数矩阵
#     k_theta[0] = 1 / np.matmul(strength_list[0], f_D.T)
#     h_theta[0] = np.multiply(k_theta[0], np.multiply(strength_list[0], f_D))
#
#     Tao = 16 * np.pi * np.power(n_surrounding, 2) * Kb * T * np.power(np.sin(theta_list[0] / 2), 2) / \
#           (3 * viscosity * lambda0 ** 2)  # 散射矢量的模
#     y = []
#     for tt in t:
#         yy = np.multiply(h_theta[0], np.exp(-Tao * tt / D_list))
#         yy = np.sum(yy)
#         y.append(yy)
#     plt.semilogx(t, np.array(y), color=color_map[index], linewidth=1, label=str(Dg) + "nm")
#     plt.xlabel("delay:t(s)", fontsize=13)
#     plt.ylabel("f(x)", fontsize=13)
#     plt.title("single peak distribute DLS curve")
#     plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
# plt.show()

single = utils.single_distribute(random=False, D_=600, sigma=0.15, start=300, end=900, count=51)
D_list, f_D = single.fetch()  # 获取仿真所得的粒径数据(单位是nm）和分布

k_theta = np.zeros(n)  # 存储散射角的权重系数
strength_list = np.zeros((n, D_list.shape[0]))  # 存储mie散射光强分数的二维数组
h_theta = np.zeros((n, D_list.shape[0]))  # 存储电场自相关系数的二维数组
D_theta = np.zeros(n)  # 输入GRNN神经网络中的输入向量
G_theta = np.zeros(n)  # 存储各个散射角处光强自相关函数基线值
for index0 in range(n):  # 开始对每个散射角进行遍历求解
    for index1 in range(D_list.shape[0]):
        x = 2 * np.pi * D_list[index1] * n_surrounding / lambda0  # 尺寸参数：相对于波长的球体尺寸
        # use miepython
        # s1, s2 = miepython.mie_S1_S2(sphere_index, x, np.cos(theta_list[index0]))
        s1, s2 = PyMieScatt.MieS1S2(sphere_index, x, np.cos(theta_list[index0]))
        scatter_strength = np.power(abs(s1), 2) + np.power(abs(s2), 2)
        strength_list[index0][index1] = scatter_strength
    light_summary = np.sum(strength_list[index0])  # 二维数组中axis=1是按行相加
    strength_list[index0] = np.divide(strength_list[index0], light_summary)  # 散射光强分数矩阵
    # print(strength_list)
    # if np.matmul(strength_list[index0], f_D.T).all()==np.multiply(strength_list[index0], f_D).all():
    #     print("ok")# np.multiply获得逐个相乘的数组，np.matmul获得前者的求和结果
    CIxfD = np.multiply(strength_list[index0], f_D)
    G_theta[index0] = 10E-7 * np.power(np.sum(CIxfD), 2)
    k_theta[index0] = 1 / np.sum(CIxfD)
    D_theta[index0] = np.sum(CIxfD) / \
                      np.matmul(strength_list[index0], np.divide(f_D, D_list))  # 计算输入到GRNN网络中的特征向量数值
    # print(k_theta)
    h_theta[index0] = np.multiply(k_theta[index0], CIxfD)
# print(h_theta, np.sum(h_theta, axis=1))  # 验证h_theta的累计和为1
# print(D_theta, f_D)  # 分别打印输入和输出
D_theta = np.multiply(1E6, D_theta)

t = np.logspace(np.log10(1E-5), np.log10(5), 200, endpoint=True)  # 时间间隔
plt.figure(figsize=(8, 6), dpi=100)
for i in range(n):
    Tao = 16 * np.pi * np.power(n_surrounding, 2) * Kb * T * np.power(np.sin(theta_list[i] / 2), 2) / \
          (3 * viscosity * lambda0 ** 2)  # 散射矢量的模
    y = []
    for tt in t:
        yy = np.multiply(h_theta[i], np.exp(-Tao * tt / D_list))
        yy = np.sum(yy)
        y.append(yy)
    plt.semilogx(t, np.array(y), color=color_map[i], linewidth=1, label=label_list[i])
    plt.xlabel("delay:t(s)", fontsize=13)
    plt.ylabel("f(x)", fontsize=13)
    plt.title("dynamic light scattering:")
    plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
plt.show()
