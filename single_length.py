import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scripts import GRNN


def draw_figure():
    lambda0 = 532E-9
    theta_list = [np.pi / 6, 2*np.pi / 9, 5*np.pi / 18, np.pi / 3]  # 以弧度计算
    color_map = ["red", "green", "yellow", "blue"]
    label_list = ["30°", "40°", "50°", "60°"]
    n_surrounding = 1.48
    eta = 20E-3
    Kb = 138E-25
    T = 298
    D = 590E-9
    Dt = Kb * T / 3 / np.pi / eta / D
    x = np.linspace(0, 1, 10000, endpoint=True)
    plt.figure(figsize=(8, 6), dpi=100)
    for index, theta in enumerate(theta_list):
        q = 4 * np.pi * n_surrounding * np.sin(theta) / lambda0  # 散射矢量的模
        Tao = Dt * q ** 2
        y = np.exp(-Tao * x)
        # plt.subplot(2,2,index+1)
        plt.semilogx(x, y, color=color_map[index], linewidth=1.0, label=label_list[index])  # marker="^",dashes=[2,1]
        plt.xlabel("x", fontsize=13)
        plt.ylabel("f(x)", fontsize=13)
        plt.title("single particle multi-angle DLS curve")
        plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
    # plt.savefig("result.png")
    plt.show()


def dataset():
    lambda0 = 633E-9
    theta_list = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 3 * np.pi / 4]  # 以弧度计算
    n_surrounding = 1.33
    eta = 89E-5
    Kb = 138E-25
    T = 298
    # D_list = [40E-8, 50E-8, 60E-8, 70E-8]
    D_list = [10E-9 * i for i in range(20, 81, 2)]
    x = np.array([10E-6, 10E-5, 10E-4, 10E-3, 10E-2])
    datas = []
    targets = []
    for D in D_list[:] * 10:
        y = []
        for theta in theta_list:
            Dt = Kb * T / 3 / np.pi / eta / D
            q = 4 * np.pi * n_surrounding * np.sin(theta) / lambda0  # 散射矢量的模
            Tao = Dt * q ** 2
            y = list(np.exp(-Tao * x)) + y
        target = D * 10E5
        datas.append(y)
        targets.append(target)
    return np.array(datas), np.array(targets).reshape(-1, 1)


def grnn_api(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    grnn = GRNN(std=0.05, verbose=True)
    grnn.train(x_train, y_train)
    print(x_train.shape, y_train.shape)
    y_predicted = grnn.predict(x_train)
    # mse = np.mean((y_predicted - y_train) ** 2)
    # print(mse)
    print(y_train[0:5], y_predicted[0:5])
    mean_error = abs(np.mean(y_predicted) - np.mean(y_train))
    mse = np.sqrt(np.mean(np.power(y_train - y_predicted, 2)))  # 均方根误差
    print("the mean error is %f,the mse error is %f" % (mean_error, mse))
    plt.figure(figsize=(8, 6), dpi=100)
    x = np.linspace(0, len(x_train), len(x_train), endpoint=True)

    plt.scatter(x, y_predicted, color="red", linewidth=1.0, label="predict")  # marker="^",dashes=[2,1]
    plt.scatter(x, y_train, color="green", linewidth=1.0, label="true")
    plt.xlabel("number of samples", fontsize=13)
    plt.ylabel("values of predicted and True", fontsize=13)
    plt.title("Performance in particle size inversion of GRNN")
    plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
    # plt.savefig("result1.jpg")
    plt.show()


if __name__ == '__main__':
    draw_figure()
    # dataset_x, dataset_y = dataset()
    # grnn_api(dataset_x, dataset_y)
