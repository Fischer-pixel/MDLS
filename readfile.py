import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
plt.rc('text', usetex=True)  # 启用对 latex 语法的支持


def read_fit():
    with open("./1.fit", "r", encoding="utf-8") as f:
        li = f.readlines()
    df = pd.read_csv("./1.fit", sep=",", skiprows=4, names=["timestamp", "true_value", "fit_value"])
    x = df["timestamp"]*1E-6
    optical_true = df["true_value"]
    optical_fit = df["fit_value"]
    beta = optical_fit[0]  # 仪器系数
    electric_fit = np.sqrt(optical_fit/beta)
    # electric_fit1 = np.sqrt(optical_fit)
    # print()
    plt.figure(figsize=(8, 6), dpi=100)
    plt.semilogx(x, optical_true, color="green", linewidth=1, label=r"$true\ optical\ curve$")
    plt.semilogx(x, optical_fit, color="red", linewidth=1, label=r"$fit\ optical\ curve$")
    plt.semilogx(x, electric_fit, color="black", linewidth=1, label=r"$electric\ curve$")
    plt.title(r"$acf\ curve$")
    plt.xlabel(r"$delay:\tau$")
    plt.ylabel("$f(x)$")
    plt.legend()
    plt.show()
    return x, electric_fit


def read_log():
    df = pd.read_csv("2log.txt", sep=",", skiprows=0)
    print(list(df.columns.values))
    x = df["d(nm)"]
    y_true = df[" G(d)"]
    diameter = x[sg.argrelmax(np.array(y_true))[0]]
    print(diameter)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(x, y_true, color="green", linewidth=1, label=r"$predict$")
    plt.title(r"$PSD$")
    plt.xlabel(r"$diameter\phi(nm)$")
    plt.ylabel("$f(x)$")
    plt.legend()
    plt.show()


def read_contin():
    df = pd.read_csv("1contin.txt", sep=",", skiprows=0, names=["D", "f_D", "accumulate"])
    x = df["D"]
    y_true = df["f_D"]
    y_fit = df["accumulate"]
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(x, y_true, color="green", linewidth=1, label=r"$distribute\ curve$")
    plt.plot(x, y_fit, color="red", linewidth=1, label=r"$accumulate\ curve$")
    plt.title(r"$distribute\ curve$")
    plt.xlabel(r"$diameter:d$")
    plt.ylabel("$f(x)$")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    read_fit()
    read_log()
    read_contin()
