<div align="center">

<h1>MDLS-model-building</h1>
本科毕业设计-多角度动态光散射建模与仿真<br><br>

</div>

------

## 简介
本仓库具有以下特点:
+ 具有软件操作界面；
+ 即便在相对较差的显卡上也能快速训练和推理;
+ 使用少量数据进行训练也能得到较好结果;
+ 可以个性化调节参数;
+ 具有多种内置的机器学习算法;
+ 实现了多维输出的GRNN网络，且计算速度可与matlab内置的newgrnn媲美。
+ 可使用sklearn和sko进行参数寻优，大家放心使用。
## 环境配置
我们推荐你使用anaconda来配置环境。

以下指令需在Python版本大于3.6的环境当中执行:
```bash
# 安装Pytorch及其核心依赖，若已安装则跳过
pip install scikit-learn scipy numpy matplotlib pandas PyMieScat

```

你也可以通过pip来安装依赖：

**注意**: `也可使用`requirements.txt`来安装依赖`

```bash
pip install -r requirements.txt
```

## 其他预模型准备
也提供了其他的一些数据集来进行推理和训练。

你可以从我们的[百度网盘](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)下载到这些模型。

以下是一份清单，包括了所有RVC所需的预模型和其他文件的名称:

之后使用以下指令来调用ui界面:
```bash
python main.py
```

我将推出一个英文版本的readme.

仓库内还有一份`简易教程.doc`以供参考。
