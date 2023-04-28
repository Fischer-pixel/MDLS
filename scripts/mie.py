import math
import miepython
import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use("seaborn-dark")

delta_t = 1E-5  # 时间间隔
num_sphere = 100
Dd = 138E-16 * 298.15 / 3 / np.pi / 89E-11 / 590  # 颗粒扩散系数
print("D=%f" % Dd)


def single_sphere_mie_scatter():
    surrounding_index = 1.48  # 介质折射率
    sphere_index = 2.63  # 球体折射率
    radius_list = [1E-9*i for i in range(100,1001,100)]  # 半径
    num_angle = 9000  # 在0-π中要求的点数
    light_lambda = 532E-9 / surrounding_index  # 光在介质中的波长
    raw_angle_array = np.array([i / num_angle for i in range(0, num_angle, 10)])
    angle_array = 180 * raw_angle_array
    radian_array = np.pi * raw_angle_array

    # 创建一个8x6大小的图像, dpi=80表示分辨率每英尺80点
    plt.figure(figsize=(8, 6), dpi=100)
    for radius in radius_list:
        x = 2 * np.pi * radius / light_lambda  # 尺寸参数：相对于波长的球体尺寸

        s1, s2 = miepython.mie_S1_S2(sphere_index/surrounding_index, x, np.cos(radian_array))
        scatter_strength = np.power(abs(s1), 2) + np.power(abs(s2), 2)
        plt.plot(angle_array, scatter_strength, color="red", linewidth=1.0, label=str(radius))  # marker="^"

        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("scattering strength with scattering angles")
        plt.legend()  # 显示图例
        # plt.savefig("result.png")
        plt.show()


def multi_sphere_DLS(angle=10):
    '''
    散射场振幅是散射体积内所有散射体复振幅的叠加量
    '''
    m = 1.5
    x = np.pi / 3
    theta = np.linspace(-180, 180, 1800)
    mu = np.cos(theta / 180 * np.pi)
    s1, s2 = miepython.mie_S1_S2(m, x, mu)
    scat = 5 * (abs(s1) ** 2 + abs(s2) ** 2) / 2  # unpolarized scattered light

    N = 13
    xx = 3.5 * np.random.rand(N, 1) - 1.5
    yy = 5 * np.random.rand(N, 1) - 2.5

    plt.scatter(xx, yy, s=40, color='red')
    for i in range(N):
        plt.plot(scat * np.cos(theta / 180 * np.pi) + xx[i], scat * np.sin(theta / 180 * np.pi) + yy[i], color='red')

    plt.plot([-5, 7], [0, 0], ':k')

    plt.annotate('incoming\nirradiance', xy=(-4.5, -2.3), ha='left', color='blue', fontsize=14)
    for i in range(6):
        y0 = i - 2.5
        plt.annotate('', xy=(-1.5, y0), xytext=(-5, y0), arrowprops=dict(arrowstyle="->", color='blue'))

    plt.annotate('unscattered\nirradiance', xy=(3, -2.3), ha='left', color='blue', fontsize=14)
    for i in range(6):
        y0 = i - 2.5
        plt.annotate('', xy=(7, y0), xytext=(3, y0), arrowprops=dict(arrowstyle="->", color='blue', ls=':'))

    # plt.annotate('scattered\nspherical\nwave', xy=(0,1.5),ha='left',color='red',fontsize=16)
    # plt.annotate('',xy=(2.5,2.5),xytext=(0,0),arrowprops=dict(arrowstyle="->",color='red'))
    # plt.annotate(r'$\theta$',xy=(2,0.7),color='red',fontsize=14)
    # plt.annotate('',xy=(2,2),xytext=(2.7,0),arrowprops=dict(connectionstyle="arc3,rad=0.2", arrowstyle="<->",color='red'))

    plt.xlim(-5, 7)
    plt.ylim(-3, 3)
    plt.axis('off')
    plt.show()


def single_sphere_mie_scatter1():
    surrounding_index = 1.48  # 介质折射率
    sphere_index = 2.63  # 球体折射率
    radius_list = [1E-9 * i for i in range(100, 1001, 100)]  # 半径
    num_angle = 9000  # 在0-π中要求的点数
    light_lambda = 532E-9 / surrounding_index  # 光在介质中的波长
    raw_angle_array = np.array([i / num_angle for i in range(0, num_angle, 10)])
    angle_array = 180 * raw_angle_array
    radian_array = np.pi * raw_angle_array

    # 创建一个8x6大小的图像, dpi=80表示分辨率每英尺80点
    plt.figure(figsize=(8, 6), dpi=100)
    for radius in radius_list:
        scatter_strength_list = []
        for radian in radian_array[:]:
            x = 2 * np.pi * radius / light_lambda  # 尺寸参数：相对于波长的球体尺寸
            s1, s2 = PyMieScatt.MieS1S2(sphere_index/surrounding_index, x, np.cos(radian))
            scatter_strength = np.power(abs(s1), 2) + np.power(abs(s2), 2)
            scatter_strength_list.append(scatter_strength)
        plt.plot(angle_array, scatter_strength_list, color="red", linewidth=1.0, label=str(int(1E9*radius))+"nm")  # marker="^"

        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("scattering strength with scattering angles")
        plt.legend()  # 显示图例
        # plt.savefig("result.png")
        plt.show()


if __name__ == '__main__':
    # multi_sphere_DLS()
    single_sphere_mie_scatter1()
    # generate_brown()
