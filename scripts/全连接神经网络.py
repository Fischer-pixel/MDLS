import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math, random
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scripts import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import seaborn as sns
from tqdm import tqdm
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=4, out_features=100, bias=True)
        self.hidden3 = nn.Linear(100, 100)
        self.predict = nn.Linear(100, 51)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        output = self.predict(x)
        return output


if __name__ == '__main__':

    # data_x,data_y = make_regression(n_samples=100,n_features=4,n_informative=5,n_targets=51,random_state=1)
    single = True
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_x = np.load(f"../data/{name}/feature.npy", encoding="latin1")
    data_y = np.load(f"../data/{name}/target.npy", encoding="latin1")
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=0.80)
    # print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    # scale = StandardScaler()
    # train_x, test_x = scale.fit_transform(train_x), scale.transform(test_x)
    # print(train_x[0],test_x[0])
    #  数据转化为张量

    train_x = torch.from_numpy(train_x.astype(np.float32))
    test_x = torch.from_numpy(test_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))
    test_y = torch.from_numpy(test_y.astype(np.float32))

    #  将训练数据处理为数据加载器
    train_data = Data.TensorDataset(train_x, train_y)
    test_data = Data.TensorDataset(test_x, test_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=80, shuffle=True, num_workers=1)

    mlp = MLP()
    mlp = mlp.to(device)
    print(mlp)

    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)
    loss = nn.MSELoss()
    train_loss_all = []
    epochs = 150
    pbar = tqdm(range(epochs), total=epochs, leave=True, ncols=100, unit="个", unit_scale=False, colour="red")
    for epoch in pbar:
        train_loss = 0
        train_num = 0
        start = time.perf_counter()
        for step, (b_x, b_y) in enumerate(train_loader):
            output = mlp(b_x.to(device))
            los = loss(output, b_y.to(device))
            optimizer.zero_grad()
            los.backward()
            optimizer.step()
            train_loss = train_loss + los.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)

        end = time.perf_counter()
        pbar.set_description(f"Epoch {epoch}/{epochs}")
        pbar.set_postfix({"正在处理": epoch}, loss=(train_loss / train_num), cost_time=(end-start))

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train_loss_all, color="red", linewidth=1, label="train loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    torch.save(mlp, f"../data/{name}/mlp.pt")
    mlp = torch.load(f"../data/{name}/mlp.pt")
    mlp = mlp.to(device)
    mlp.eval()

    #  预测
    with torch.no_grad():
        t1 = time.perf_counter()
        predict_y = mlp(test_x.to(device))
        t2 = time.perf_counter()
        print(f"It spends {(t2 - t1)/test_x.shape[0]} seconds to inference one sample")
        predict_y = predict_y.cpu().numpy()
        # 带上噪声污染的反演
        noise_single = torch.multiply(0.01 * torch.mean(test_x), torch.randn((test_x.shape[1])))
        x_test_noise = test_x + noise_single
        print(x_test_noise[0],test_x[0])
        y_predict_noise = mlp(x_test_noise.to(device)).cpu().numpy()

        # 预测结果汇总输出
        V = utils.getV(test_y.numpy(), predict_y)
        print(f"the mean performance param is {np.mean(V)}")
        t = np.linspace(300, 900, test_y.shape[1], endpoint=True)  # X轴坐标
        plt.figure(figsize=(8, 6), dpi=100)
        test_shuffle = np.random.randint(0, test_x.shape[0], 5)  # 从测试集中随机抽取5个样本出来可视化结果
        for i in test_shuffle:  # 看测试集上前5个的预测情况
            Dg_estimate = utils.get_line(np.arange(300, 901, step=(900 - 300) / (51 - 1)),
                                         predict_y[i], test_y.numpy()[i], single)
            print(Dg_estimate)
            plt.scatter(t, test_y.numpy()[i], color="green", linewidth=1, label="True")
            plt.scatter(t, y_predict_noise[i], color="black", linewidth=1, label="predict with noise")
            plt.scatter(t, predict_y[i], color="red", linewidth=1, label="predict")
            plt.xlabel("粒径范围(nm)", fontsize=13)
            plt.ylabel("粒径分布", fontsize=13)
            plt.title("使用全连接神经网络预测粒径分布")
            plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
            plt.show()

