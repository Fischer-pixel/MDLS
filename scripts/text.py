import PyMieScatt
import miepython
import numpy as np
from tqdm import tqdm
import cv2, random
import time, torch
from sklearn.model_selection import train_test_split
import scipy.signal as sg
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl

a = np.array([1, 2, 3])
a = np.mat(a)
print(a)
b = np.array([1, 2, 3])
c, d = np.matmul(a, b.T), np.multiply(a, b)

print(np.divide(a, b), np.multiply(2, a), c, a + b)

b = np.arange(200, 801, step=20)
c = np.zeros(5)
t = np.logspace(np.log10(1E-8), np.log10(5), 200, endpoint=True)  # 时间间隔
print(t)
a = np.array([0, 1, 2, 3, 4, 5, 4, 3, 5, 7, 8, 7, 6, 4, 2, 1, 0])
print(sg.argrelmax(a), sg.argrelmin(a))


a = (1,2)
print("%.2f %.2f"%a)
# 极小值 的下标
def numpy2matlab():
    single = False
    name = "single" if single else "multiply"
    data_x = np.load(f"../data/oil/multi-angle/{name}/dataset/feature.npy", encoding="latin1")
    # data_y = np.load(f"../data/{name}/line_target.npy", encoding="latin1")
    print(data_x.shape)
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.8, random_state=42)
    # np.save(f"../data/{name}/train/feature_one_angle.npy", x_train)
    # np.save(f"../data/{name}/train/target_one_angle.npy", y_train)
    # np.save(f"../data/{name}/test/feature_one_angle.npy", x_test)
    # np.save(f"../data/{name}/test/target_one_angle.npy", y_test)
    # # 将numpy数组转化为matlab可读取的mat文件格式
    # from scipy import io
    #
    # io.savemat(f'../matlab/{name}/train/feature.mat', {'x': x_train})
    # io.savemat(f'../matlab/{name}/train/target.mat', {'y': y_train})
    # io.savemat(f'../matlab/{name}/test/feature.mat', {'x': x_test})
    # io.savemat(f'../matlab/{name}/test/target.mat', {'y': y_test})


def processing(iterable_object):
    pbar = tqdm(iterable_object,
                total=len(iterable_object),
                leave=True,
                ncols=100,
                unit="个",
                unit_scale=False,
                colour="blue")
    '''iterable: 可迭代的对象, 在⼿动更新时不需要进⾏设置
    desc: 字符串, 左边进度条描述⽂字
    total: 总的项⽬数
    leave: bool值, 迭代完成后是否保留进度条
    file: 输出指向位置, 默认是终端, ⼀般不需要设置
    ncols: 调整进度条宽度, 默认是根据环境⾃动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
    unit: 描述处理项⽬的⽂字, 默认是it, 例如: 100 it/s, 处理照⽚的话设置为img ,则为 100 img/s
    unit_scale: ⾃动根据国际标准进⾏项⽬处理速度单位的换算, 例如 100000 it/s >> 100k it/s
    colour: 进度条颜色'''
    for idx, element in enumerate(pbar):
        time.sleep(0.5)
        pbar.set_description(f"Epoch {idx}/{len(iterable_object)}")
        pbar.set_postfix({"正在处理的元素为": element}, loss=random.random(), cost_time=random.randrange(0, 100))
    '''
    with tqdm.tqdm(total=10) as bar:  # total为进度条总的迭代次数
        # 操作1
        time.sleep(1)
        # 更新进度条
        bar.update(1)  # bar.update()里面的数表示更新的次数，和optimizer.step方法类似
    '''


def photo():
    # 调用usb摄像头
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))  # 要重设
    # 显示
    while True:
        ret, frame = cap.read()
        cv2.imshow("window1", frame)
        print("ok")
        if cv2.waitKey(0) or 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def use_latex():
    plt.rc('text', usetex=True)  # 启用对 latex 语法的支持
    x = np.linspace(0, 1, 50, endpoint=True)
    y = np.sin(x)
    plt.plot(x, y, color="red", linewidth=1, label=r"$\Lambda$")
    plt.title(r"$\sigma$")
    plt.legend()
    plt.show()


def jpg2ico():
    import PythonMagick

    img = PythonMagick.Image("../img.png")  # 加载需要转换的图片
    img.sample('32x32')  # 设置生成的 icon 的大小
    img.write("../logo.ico")  # 生成 icon 并保存


if __name__ == '__main__':
    # numpy2matlab()
    use_latex()
    # jpg2ico()
    x = np.array([0, 1, 2, 3, 4, 5, 4, 3, 5, 7, 8, 7, 6, 4, 2, 1, 0])
    if x is None:
        x = np.logspace(np.log10(1E-5), np.log10(5), 50, base=10.0, endpoint=True)
