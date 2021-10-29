#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt


from model.network import network
from model.dataset import dataset
from model.activation_layer import RELU,GELU,Mish,Swish,change_dim
from utils.train import Runthread
from utils.draw import show_layers
device = "cuda" if torch.cuda.is_available() else "cpu"

class graphic_interface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_main = QFormLayout()
        

        font = QFont()
        font.setPointSize(20)   #括号里的数字可以设置成自己想要的字体大小
        self.setWindowTitle('network_visualization')       


        self.network = network()
        self.conv = ConvNet_arguments(self.network,self)
        self.linear = linear_arguments(self.network,self)
        self.act  = activation_arguments(self.network,self)
        self.pool = pool_arguments(self.network,self)
        self.reshape = change_dim

        #the default dataset is MNIST
        self.dataset = 'MNIST'
        # self.channel_ = 1
        # self.height_ = 28
        # self.width_ = 28

        self.button_add_conv = QPushButton()
        self.button_add_conv.clicked.connect(self.add_conv)   #按钮的关联函数
        self.button_add_conv.setText("添加卷积层")                    
        
        self.button_add_pool = QPushButton()
        self.button_add_pool.clicked.connect(self.add_pool)
        self.button_add_pool.setText("添加池化层")

        self.button_add_dropOut = QPushButton()
        self.button_add_dropOut.clicked.connect(self.add_dropOut)
        self.button_add_dropOut.setText("添加dropout层")

        self.button_change_dim = QPushButton()
        self.button_change_dim.clicked.connect(self.change_dim)
        self.button_change_dim.setText("改变维度")

        self.button_add_linear = QPushButton()
        self.button_add_linear.clicked.connect(self.linear.handle_show)
        self.button_add_linear.setText("添加全连接层")
        
        self.button_add_activation = QPushButton()
        self.button_add_activation.clicked.connect(self.act.handle_show)
        self.button_add_activation.setText("添加激活层")


        self.button_delete = QPushButton()
        self.button_delete.clicked.connect(self.delete)
        self.button_delete.setText("清空")

        self.button_dataset = QComboBox(self)
        self.button_dataset.addItems(['MNIST','Fashion_MNIST'])
        self.button_dataset.currentIndexChanged[str].connect(self.dataset_select) # 条目发生改变，发射信号，传递条目内容
        

        self.button_start  = QPushButton()
        self.button_start.clicked.connect(self.start_training)
        self.button_start.setText("开始训练")

        self.tab = QTabWidget()
        self.network_structure = QPlainTextEdit()
        self.train_process = QLabel()
        self.tab.addTab(self.network_structure,'网络结构')
        self.tab.addTab(self.train_process,'测试结果')

        self.layout_main.addWidget(self.button_add_conv)
        self.layout_main.addWidget(self.button_add_pool)
        self.layout_main.addWidget(self.button_add_dropOut)
        self.layout_main.addWidget(self.button_add_linear)
        self.layout_main.addWidget(self.button_change_dim)
        self.layout_main.addWidget(self.button_add_activation)
        self.layout_main.addWidget(self.button_delete)
        self.layout_main.addRow("数据集:",self.button_dataset)
        self.layout_main.addWidget(self.button_start)


        hbox = QHBoxLayout()
        hbox.addLayout(self.layout_main)

        hbox.addWidget(self.tab)
        self.show_structure()

        self.setLayout(hbox)

        self.dataset_select(self.dataset)

    def add_conv(self):
        self.conv.handle_show()

    def add_pool(self):
        self.pool.handle_show()

    def add_dropOut(self):
        value, ok = QInputDialog.getDouble(self, "dropout", "请输入P:", 0.5,-1,1)
        if ok==False:
            return
        new_Dropout_layer = nn.Dropout2d(p=value)
        self.network.layers.append(new_Dropout_layer)
        self.network.layers_arguments.append({'dropout':value})
        QMessageBox.about(self, "提示", '已成功添加dropout层')
        self.show_structure()

    def change_dim(self):
        value, ok = QInputDialog.getText(self, "dimension", "请输入改变后的维度,用\'x\'相连",QLineEdit.Normal, "-1x320")
        if ok ==False:
            return
        dim = value.split("x")
        dim = [int(i) for i in dim]
        self.network.layers.append(self.reshape(dim))
        self.network.layers_arguments.append({'change_dim':dim[-1]})
        QMessageBox.about(self, "提示", '向量维度会在此改变')
        self.show_structure()

    def delete(self):
        self.network = network(self.channel_,self.height_,self.width_)
        self.show_structure()
        QMessageBox.about(self, "提示", '神经网络已重置')

    def dataset_select(self,i):
        self.dataset = i
        self.thread_dataset = dataset(self.dataset)
        self.thread_dataset._signal.connect(self.call_back)
        self.thread_dataset.start()
      
    def call_back(self,msg):
        self.train_dataloader = msg[0]
        self.test_dataloader = msg[1]
        self.channel_ = msg[2]
        self.height_ = msg[3]
        self.width_ = msg[4]
        self.test_data = msg[5]
        self.label_map = msg[6]
        self.network.layers_arguments = [{'orgin':msg[2:5]}]
        self.show_structure()
        QMessageBox.about(self, "提示", '该数据集有10类分类结果,\n神经网络最后的输出结点数应为10')

    def show_structure(self):
        layers =show_layers(self.network.layers_arguments)
        self.network_structure.setPlainText(layers)
        
    def show_train_process(self,model):
        figure = plt.figure(figsize=(10, 10))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.test_data), size=(1,)).item()
            img, label = self.test_data[sample_idx]
            with torch.no_grad():
                img=img.unsqueeze(0).to(device)
                pre = model(img)
                target = pre.argmax(1)
            img = img.to('cpu')
            figure.add_subplot(rows, cols, i)
            plt.title('True label: '+self.label_map[label]+'\n'+'result :'+self.label_map[(target.cpu().item())])
            plt.axis("off")
            plt.imshow(img.squeeze())
        plt.savefig("train_process.jpg")

    def update_pic(self):
        jpg = QPixmap("train_process.jpg").scaled(self.train_process.width(), self.train_process.height())
        self.train_process.setPixmap(jpg)

    def start_training(self):
        QMessageBox.about(self, "提示", '此程序不会检查神经网络链接的正确性')
        self.network.link_layer()
        print(self.network.sequential)
        self.network.to(device)
        self.thread_train = Runthread(self.network,self.train_dataloader,self.test_dataloader,self)
        self.thread_train.start()

class ConvNet_arguments(QWidget):
    def __init__(self,network,graph):
        super().__init__()
        self.layout = QFormLayout()
        self.network = network
        self.graph = graph

        self.input_channel = QLineEdit("1")
        self.output_channel = QLineEdit("15")
        self.kernel_size = QLineEdit("5")
        self.padding = QLineEdit("0")
        self.button = QPushButton()
        self.button.clicked.connect(self.handle_close)
        self.button.setText("确认")

        self.layout.addRow("输入通道个数 inputchannel:",self.input_channel)
        self.layout.addRow("输出通道个数 outputchannel:",self.output_channel)
        self.layout.addRow("卷积核大小 kernel_size:",self.kernel_size)
        self.layout.addRow("填充个数:",self.padding)
        self.layout.addRow(self.button)
        self.setLayout(self.layout)

    def handle_show(self):
        self.show()

    def handle_close(self):
        new_ConvNet_layer = nn.Conv2d(int(self.input_channel.text()),
                                    int(self.output_channel.text()),
                                    kernel_size=int(self.kernel_size.text()),
                                    padding=int(self.padding.text())
                                    )
        self.network.layers.append(new_ConvNet_layer)
        self.network.layers_arguments.append({'conv':[int(self.output_channel.text()),int(self.kernel_size.text()),int(self.kernel_size.text())]})
        QMessageBox.about(self, "提示", '已成功添加卷积层')
        self.graph.show_structure()
        self.close()
        
class linear_arguments(QWidget):
    def __init__(self,network,graph):
        super().__init__()
        self.layout = QFormLayout()
        self.network = network
        self.graph = graph

        self.input_channel = QLineEdit("784")
        self.output_channel = QLineEdit("16")
        self.button = QPushButton()
        self.button.clicked.connect(self.handle_close)
        self.button.setText("确认")

        self.layout.addRow("输入通道个数 inputchannel:",self.input_channel)
        self.layout.addRow("输出通道个数 outputchannel:",self.output_channel)
        self.layout.addRow(self.button)
        self.setLayout(self.layout)

    def handle_show(self):
        self.show()

    def handle_close(self):
        new_linear_layer = nn.Linear(int(self.input_channel.text()), int(self.output_channel.text()))
        self.network.layers.append(new_linear_layer)
        self.network.layers_arguments.append({'linear':[int(self.input_channel.text()),int(self.output_channel.text())]})
        QMessageBox.about(self, "提示", '已成功添加linear层')
        self.graph.show_structure()
        self.close()
        
class activation_arguments(QWidget):
    def __init__(self,network,graph):
        super().__init__()
        self.layout = QFormLayout()

        self.network = network
        self.graph = graph

        self.activation =RELU()
        self.type = QComboBox()
        self.type_name = 'RELU'
        
        self.type.addItems(['RELU','Mish','Swish','GELU'])
        self.type.currentIndexChanged[str].connect(self.type_select) # 条目发生改变，发射信号，传递条目内容
        self.button = QPushButton()
        self.button.clicked.connect(self.handle_close)
        self.button.setText("确认")

        self.layout.addRow("激活函数选择:",self.type)
        self.layout.addRow(self.button)
        self.setLayout(self.layout)

    def type_select(self,i):
        self.type_name = i
        KW = {'RELU':RELU,'Mish':Mish,'Swish':Swish,'GELU':GELU}
        self.activation = KW[i]()

    def handle_show(self):
        self.show()

    def handle_close(self): 
        self.network.layers.append(self.activation)
        self.network.layers_arguments.append({'activation':self.type_name})
        QMessageBox.about(self, "提示", '已成功添加激活函数')
        self.graph.show_structure()
        self.close()
        
class pool_arguments(QWidget):
    def __init__(self,network,graph):
        super().__init__()
        self.layout = QFormLayout()
        self.network = network
        self.graph = graph

        self.pool = nn.MaxPool2d
        self.type = QComboBox()
        
        self.type.addItems(['Maxpool','Avgpool'])
        self.type.currentIndexChanged[str].connect(self.type_select) # 条目发生改变，发射信号，传递条目内容
        self.button = QPushButton()
        self.button.clicked.connect(self.handle_close)
        self.button.setText("确认")

        self.layout.addRow("池化层选择:",self.type)
        self.layout.addRow(self.button)
        self.setLayout(self.layout)

    def type_select(self,i):
        KW = {'Maxpool':nn.MaxPool2d,'Avgpool':nn.AvgPool2d}
        self.pool = KW[i]

    def handle_show(self):
        self.show()

    def handle_close(self):
        kernel_size = 2
        self.new_pool_layer = self.pool(kernel_size=kernel_size)
        self.network.layers.append(self.new_pool_layer)
        self.network.layers_arguments.append({'pool':kernel_size})
        QMessageBox.about(self, "提示", '已成功添加池化层')
        self.graph.show_structure()
        self.close()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    tool = graphic_interface()
    tool.show()
    sys.exit(app.exec_())