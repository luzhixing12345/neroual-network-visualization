# 神经网络结果可视化

## 介绍：

### 组件介绍：

- 界面：PYQT5+pytorch实现

![image-20211026224106512](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224106512.png)

- 卷积层选择

![image-20211026224250738](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224250738.png)

- 池化层选择

![image-20211026230107202](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230107202.png)

- 激活函数选择

![image-20211026224406505](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224406505.png)

- dropout 

  ![image-20211026231747465](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026231747465.png)
- 全连接层

![image-20211026230135107](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230135107.png)

- 改变维度

![image-20211026230203884](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230203884.png)

- 数据集切换

![image-20211026230233800](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230233800.png)

- 网络定义实例

![image-20211026224608957](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224608957.png)



### MNIST+EXAMPLE_NET1训练实例：

EPOCH = 0 <img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224811860.png" alt=" " style="zoom:80%;" />



EPOCH = 5 <img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224921794.png" style="zoom:80%;" />

EPOCH = 10<img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026225138409.png" style="zoom:80%;" />

![](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026225344354.png) ![image-20211026225323845](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026225323845.png)



### Fashion_MNIST+EXAMPLE_NET1训练实例

<img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029110821938.png" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029110858665.png" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029110925611.png" style="zoom:50%;" />

## 使用方法：

### requirements:

- pytorch
- matplotlib
- pyqt5



### run:

```shell
python graphical_interface.py
```



### attention:

- 本项目提供了神经网络层的定义，需要自行传参
- 不会检查是否各层之间连接是否正确，以及是否向量大小是否符合batch_size
- 本项目提供了两个实例网络，在model/network.py中，如没有很好的构建神经网络基础可模拟该网络实现

- 第一次下载数据集时需要联网

- popUpWindows.py没有用到，是一个很好的pyqt5做弹出窗口的代码示例







## 代码实现:

- graphic_interface 主界面函数实现了添加各层之间的部分，以及清空和切换

![image-20211029111418532](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029111418532.png) 

- 卷积 全连接 激活函数 池化分别用四个模块实现，包含内部参数选择以及返回值输出

![image-20211029111628599](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029111628599.png) 

- 四个高效的激活函数的内部实现 model/activation.py

![image-20211029111903424](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029111903424.png) 

- 数据集的选择以及数据集对应的标签 model/dataset.py

![image-20211029112034864](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029112034864.png) 

- 神经网络基类以及两个案例神经网络结构 model/network.py

![image-20211029112122844](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029112122844.png) 

- 神经网络结构 文字格式可视化 utils.draw/py

说实话这里我原来是打算也做成图片格式的可视化的，但是我尝试了latex matplotlib pyqtgraph 等可视化图形库，依然没有做到令我满意的图片，实在是太丑陋了。。。。

索性就把这部分注销掉了，直接变成了文字格式了

如后续有兴趣继续尝试可以继承工作继续进展

![image-20211029112325715](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029112325715.png) 

- 训练+测试+一个线程类的实现 utils/train.py

如果不用多线程的话十分卡顿，电脑处理器不够

这样训练测试阶段就可以直接多线程处理，与本身的exec_()并行

![image-20211029112720823](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211029112720823.png) 

