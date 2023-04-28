import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from scipy.optimize import brenth#  找到函数在区间的根，
from scipy.integrate import odeint#odeint()函数需要至少三个变量，第一个是微分方程函数，第二个是微分方程初值，第三个是微分的自变量
from scipy.interpolate import InterpolatedUnivariateSpline#将散点图拟合成平滑的曲线（函数）


def train():
    iris = datasets.load_iris()
    x = iris.data[:, :]  # 选取前两列作为X参数
    y = iris.target  # 采集标签作为y参数
    print(x[:5], y[:5])
    model = svm.SVC(C=5, kernel='poly', gamma=5, probability=True)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    model.fit(train_x, train_y)
    #joblib.dump(model, "train.pkl")

    # 建图
    def pic():
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        h = (x_max / x_min) / 100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        plt.subplot(1, 1, 1)  # 将显示界面分割成1*1 图形标号为1的网格
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # np.c_按行(行数相等）,np.r_按列连接两个矩阵,但变量为两个数组，按列连接
        Z = Z.reshape(xx.shape)  # 重新构造行列
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)  # 绘制等高线
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)  # 生成一个scatter散点图。
        plt.xlabel('Sepal length')  # x轴标签
        plt.ylabel('Sepal width')  # y轴标签
        plt.xlim(xx.min(), xx.max())  # 设置x轴的数值显示范围
        plt.title('SVC with linear kernel')  # 设置显示图像的名称
        plt.savefig('./test1.png')  # 存储图像
        plt.show()  # 显示
    predict_y = model.predict(test_x)
    print("predict:", predict_y)
    print(test_y)
    print("train precision", model.score(train_x, train_y))
    print("test precision", model.score(test_x, test_y))
