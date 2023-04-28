import torch
import time
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import utils

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


class Net(nn.Module):
    def __init__(self, kind="RNN", input_dim=4, hidden_dim=100, layer_dim=3, out_dim=51):
        super(Net, self).__init__()
        if kind == "RNN":
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim,
                              batch_first=True, nonlinearity="relu")
        elif kind == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim,
                               batch_first=True, bias=True, dropout=0.5, bidirectional=False, proj_size=0)
        elif kind == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim,
                              batch_first=True, bias=True, dropout=0.5, bidirectional=False)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        out, h_n = self.rnn(x, None)
        out = self.fc1(out[:, -1, :])
        return out


if __name__ == '__main__':
    device = torch.device("cuda:0")
    single = True
    kind = "LSTM"
    name = "single" if single else "multiply"  # 存放数据的文件夹名字
    # train_x,train_y = make_regression(n_samples=100,n_features=10,n_informative=5,n_targets=2,random_state=1)
    x_train = np.load(f"../data/{name}/line_feature.npy", encoding="latin1")
    y_train = np.load(f"../data/{name}/line_target.npy", encoding="latin1")
    x_test = np.load(f"../data/{name}/test/feature.npy", encoding="latin1")
    y_test = np.load(f"../data/{name}/test/target.npy", encoding="latin1")
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.80, random_state=0)
    #  数据转化为张量
    train_x = torch.from_numpy(x_train.astype(np.float32)).half()
    test_x = torch.from_numpy(x_test.astype(np.float32)).half()
    train_y = torch.from_numpy(y_train.astype(np.float32)).half()
    test_y = torch.from_numpy(y_test.astype(np.float32)).half()

    #  将训练数据处理为数据加载器
    train_data = Data.TensorDataset(train_x, train_y)
    test_data = Data.TensorDataset(test_x, test_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)

    model = Net(kind).to(device).half()
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss = nn.MSELoss()
    train_loss_all = []
    epochs = 150
    pbar = tqdm(range(epochs), total=epochs, leave=True, ncols=100, unit="个", unit_scale=False, colour="red")
    for epoch in pbar:
        train_loss = 0
        train_num = 0
        start = time.perf_counter()
        for step, (b_x, b_y) in enumerate(train_loader):
            # 输入[batch,time_step,input_dim]
            b_x = b_x.view(-1, 1, 4)
            output = model(b_x.to(device))
            los = loss(output, b_y.to(device))
            optimizer.zero_grad()
            los.backward()
            optimizer.step()
            train_loss = train_loss + los.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)

        end = time.perf_counter()
        pbar.set_description(f"Epoch {epoch}/{epochs}")
        pbar.set_postfix({"正在处理": epoch}, loss=(train_loss / train_num), cost_time=(end - start))

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train_loss_all, color="red", linewidth=1, label="train loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    #  预测
    torch.save(model, f"../data/{name}/model/{kind}.pt")
    mlp = torch.load(f"../data/{name}/model/{kind}.pt")
    mlp = mlp.to(device)
    model.eval().half()
    with torch.no_grad():
        t1 = time.perf_counter()
        test_x_input = test_x.view(-1, 1, 4)
        predict_y = model(test_x_input.to(device))
        t2 = time.perf_counter()
        print(f"It spends {(t2 - t1) / test_x.shape[0]} seconds to inference one sample")
        predict_y = predict_y.cpu().numpy()

        # 带上噪声污染的反演
        noise_single = torch.multiply(0.01 * torch.mean(test_x), torch.randn((test_x.shape[1])))
        x_test_noise = test_x + noise_single
        x_test_noise_input = x_test_noise.view(-1, 1, 4)
        y_predict_noise = model(x_test_noise_input.to(device)).cpu().numpy()

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
            plt.title("使用循环神经网络预测粒径分布")
            plt.legend(loc=1, handlelength=3, fontsize=13)  # 在右上角显示图例
            plt.show()
