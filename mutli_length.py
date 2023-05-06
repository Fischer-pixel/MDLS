import time
import yaml
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import miepython
import PyMieScatt
import warnings, sys
from scripts import utils
from tqdm import tqdm
from collections import namedtuple, OrderedDict, Counter

warnings.filterwarnings("ignore")


class generator:
    def __init__(self, material="water", single=True, theta_list=None, add_noise=False,
                 draw_fig=False, method="cumulate"):
        '''
        draw_fig:  是否绘图，记得改plt.show()的位置
        method:  "line":数据集特征是电场自相关曲线,"cumulate":数据集特征是累积量计算结果
        '''
        self.material = material
        with open('config.yaml', 'r', encoding='utf-8') as f:
            data = yaml.load(stream=f, Loader=yaml.FullLoader)
            self.configs = data[self.material]  # 存储所有参数的结构体
        if theta_list is None:
            if int(self.configs["single_angle"]):
                self.theta_list = [np.pi / 4]
            else:
                self.theta_list = [np.pi / 4, np.pi / 3, np.pi / 2, 3 * np.pi / 4]  # 以弧度计算
        else:
            self.theta_list=theta_list
        self.n = len(self.theta_list)  # 输入向量的维度。后面会多次用到
        self.Tao = np.zeros(self.n)
        for i in range(self.n):
            Tao = 16 * np.pi * np.power(float(self.configs["n_surrounding"]), 2) * float(
                self.configs["Kb"]) * float(self.configs["T"]) * np.power(
                np.sin(self.theta_list[i] / 2), 2) / \
                  (3 * float(self.configs["viscosity"]) * float(self.configs["lambda0"]) ** 2)  # 散射矢量的模
            self.Tao[i] = Tao

        self.single = single
        self.add_noise = add_noise  # 训练集元素是否添加噪声
        self.draw_fig = draw_fig
        self.method = method
        self.name = "single" if self.single else "multiply"  # 存放数据的文件夹名字

    def check_single(self):
        if self.single:  # 单峰分布
            Dg_list = range(400, 901, 2)
            sigma_list = [i / 100 for i in range(5, 31)]  # range只能生成整数序列
            ratio_list = [0]
        else:  # 多峰分布
            Dg_list, sigma_list, ratio_list = [], [], []
            d_low, d_high = range(450, 550, 10), range(650, 750, 10)  # 双峰分布的两峰值位置
            sigma_big, sigma_small = [i / 100 for i in range(10, 20, 2)], [i / 1000 for i in
                                                                           range(40, 50, 2)]  # 获得两个sigma
            ratio_big = [i / 100 for i in range(70, 91, 10)]
            for low in d_low:
                for high in d_high:
                    Dg_list.append([low, high])
            for big in sigma_big:
                for small in sigma_small:
                    sigma_list.append([big, small])
            for big in ratio_big:
                ratio_list.append([big, 1 - big])
        return Dg_list, sigma_list, ratio_list

    def generate(self):
        Dg_list, sigma_list, ratio_list = self.check_single()
        data_x, data_y = [], []
        for ratio in ratio_list:
            pbar = tqdm(Dg_list, total=len(Dg_list), leave=True, ncols=150, unit="个", unit_scale=False, colour="red")
            for idx, Dg in enumerate(pbar):
                pbar.set_description(f"Epoch {idx}/{len(Dg_list)}")
                start = time.perf_counter()
                for sigma in sigma_list:
                    if self.single:
                        single = utils.single_distribute(random=False, D_=Dg, sigma=sigma,
                                                         start=100, end=1200, count=51)
                        D_list, f_D = single.fetch()  # 获取仿真所得的粒径数据(单位是nm）和分布
                    else:
                        multi = utils.multi_distribute(random=False, d_list=Dg, sigma_list=sigma,
                                                       ratio_list=ratio, start=300, end=900, count=51)
                        D_list, f_D = multi.fetch()  # 获取仿真所得的粒径数据(单位是nm）和分布

                    k_theta = np.zeros(self.n)  # 存储散射角的权重系数
                    strength_list = np.zeros((self.n, D_list.shape[0]))  # 存储mie散射光强分数的二维数组
                    h_theta = np.zeros((self.n, D_list.shape[0]))  # 存储电场自相关系数的二维数组
                    D_theta = np.zeros(self.n)  # 输入GRNN神经网络中的输入向量
                    G_theta = np.zeros(self.n)  # 存储各个散射角处光强自相关函数基线值
                    for index0 in range(self.n):  # 开始对每个散射角进行遍历求解
                        for index1 in range(D_list.shape[0]):
                            # 尺寸参数：相对于波长的球体尺寸
                            x = 2 * np.pi * D_list[index1] * float(self.configs["n_surrounding"]) / float(
                                self.configs["lambda0"])

                            # s1, s2 = miepython.mie_S1_S2(sphere_index, x, np.cos(theta_list[index0]))
                            s1, s2 = PyMieScatt.MieS1S2(
                                float(self.configs["sphere_index"]) / float(self.configs["n_surrounding"]),
                                x, np.cos(self.theta_list[index0]))
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
                                          np.matmul(strength_list[index0],
                                                    np.divide(f_D, D_list))  # 计算输入到GRNN网络中的特征向量数值
                        # print(k_theta)
                        h_theta[index0] = np.multiply(k_theta[index0], CIxfD)
                    # print(h_theta, np.sum(h_theta, axis=1))  # 验证h_theta的累计和为1
                    # print(D_theta, f_D)  # 分别打印输入和输出
                    D_theta = np.multiply(1E6, D_theta)

                    if self.method == "line":
                        t = np.logspace(np.log10(1E-5), np.log10(5), 50, base=10.0, endpoint=True)  # 时间间隔
                        buffer = []
                        for i in range(self.n):
                            y = []
                            for tt in t:
                                yy = np.multiply(h_theta[i], np.exp(-self.Tao[i] * tt / D_list))
                                yy = np.sum(yy)
                                y.append(yy)
                            buffer.append(y)
                            if self.draw_fig:
                                plt.figure(figsize=(8, 6), dpi=100)
                                plt.semilogx(t, np.array(y), color=self.configs["color_map"][i],
                                             linewidth=1, label=self.configs["label_list"][i])
                                plt.xlabel("delay:t(s)", fontsize=13)
                                plt.ylabel("f(x)", fontsize=13)
                                plt.title("dynamic light scattering:")
                                plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
                                plt.show()
                        data_x.append(buffer)
                        data_y.append(f_D)
                    else:  # method == "cumulate"
                        data_x.append(D_theta)  # 将理论值先添加到数据集中
                        data_y.append(f_D)

                end = time.perf_counter()
                pbar.set_postfix({"正在处理的中心粒径": Dg, "双峰加权系数": ratio}, cost_time=(end - start))
        data_x, data_y = np.array(data_x), np.array(data_y)
        print(f"the shape of x is {data_x.shape}, the shape of y is {data_y.shape}")
        return data_x, data_y


if __name__ == '__main__':
    pass
# print(data_y[24:27, 24:27], data_x[:3])
#
# np.save(f"./data/{name}/feature_one_angle.npy", data_x)
# np.save(f"./data/{name}/target_one_angle.npy", data_y)
