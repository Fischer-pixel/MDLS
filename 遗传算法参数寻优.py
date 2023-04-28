import numpy as np
import matplotlib.pyplot as plt
import torch
from sko.GA import GA
import time
from scripts.GRNN import GRNN
from scripts import utils
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


def function(spread):
    grnn = GRNN(sigma=spread)
    grnn.fit(X=x_train, y=y_train)
    predict_y = grnn.predict(x_test)

    # 预测结果汇总输出
    V = utils.getV(y_test, predict_y)
    return np.mean(V)


def use_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ga = GA(func=function, n_dim=1, size_pop=50, max_iter=100, lb=0.0001, ub=0.1, precision=1e-6, prob_mut=0.001)
    ga.to(device=device)
    start_time = time.perf_counter()
    best_x, best_score = ga.run()
    # generation_best_x = ga.generation_best_X()
    # generation_best_y = ga.generation_best_Y()
    end_time = time.perf_counter()
    # print(generation_best_x,"\n",generation_best_y)
    print(f"It spends {end_time - start_time} seconds to iter")
    print('the best spread factor is:', best_x, "the best score is:", best_score)


def use_cpu():
    ga = GA(func=function, n_dim=1, size_pop=50, max_iter=100, lb=0.0001, ub=0.1, precision=1e-6, prob_mut=0.001)
    start_time = time.perf_counter()
    best_x, best_score = ga.run()
    end_time = time.perf_counter()
    print(f"It spends {end_time - start_time} seconds to iter")
    print(print('the best spread factor is:', best_x, "the best score is:", best_score))


if __name__ == '__main__':
    single = False
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    x_train = np.load(f"./data/{name}/train/feature.npy", encoding="latin1")
    y_train = np.load(f"./data/{name}/train/target.npy", encoding="latin1")
    x_test = np.load(f"./data/{name}/test/feature.npy", encoding="latin1")
    y_test = np.load(f"./data/{name}/test/target.npy", encoding="latin1")
    # data_x, data_y = make_regression(n_samples=100, n_features=4, n_informative=5, n_targets=51, random_state=0)
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.90, random_state=0)
    use_gpu()
