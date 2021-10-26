# 神经网络结果可视化

## 介绍：

- 界面：PYQT5+pytorch实现

![image-20211026224106512](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224106512.png)

- 卷积层选择

![image-20211026224250738](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224250738.png)

- 池化层选择

![image-20211026230107202](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230107202.png)

- 激活函数选择

![image-20211026224406505](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224406505.png)

- dropout 

![image-20211026230029695](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230029695.png)

- 全连接层

![image-20211026230135107](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230135107.png)

- 改变维度

![image-20211026230203884](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230203884.png)

- 数据集切换

![image-20211026230233800](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026230233800.png)

- 网络定义实例

![image-20211026224608957](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224608957.png)



EPOCH = 0 <img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224811860.png" alt=" " style="zoom:80%;" />



EPOCH = 5 <img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026224921794.png" style="zoom:80%;" />

EPOCH = 10<img src="https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026225138409.png" style="zoom:80%;" />

![](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026225344354.png) ![image-20211026225323845](https://raw.githubusercontent.com/learner-lu/picbed/master/image-20211026225323845.png)



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

