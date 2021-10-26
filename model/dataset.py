
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
Dataset = {'MNIST':datasets.MNIST,'Fashion_MNIST':datasets.FashionMNIST}

MNIST_class = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

Fashion_MNIST_class = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
DATASET_CLASS = {'MNIST':MNIST_class,'Fashion_MNIST':Fashion_MNIST_class}
class dataset(QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(tuple)

    def __init__(self,D):
        super(dataset,self).__init__()
        self.dataset = D
    
    def run(self):
        kw =get_dataset(self.dataset)
        # self.train_dataloader = kw[0]
        # self.test_dataloader = kw[1]
        # self.channel = kw[2]
        # self.height = kw[3]
        # self.width = kw[4]
        self._signal.emit(kw)  # 注意这里与_signal = pyqtSignal(str)中的类型相同

def get_dataset(name):  
    
    '''
    arguments:cfg代表configure 配置文件，其中包含的信息可以从./configure.py中查找到

    return  :train_dataloader,test_dataloader分别是训练集和测试集的tensor形式
    '''
    # Download test data from open datasets.
    # 这一步运行过程中可能出现RuntimeError,这是由于网络波动问题，重复操作即可
    # 该函数会优先扫描root路径下是否存在数据集，如果已经下载过数据集则会自动跳过，不需担心重复下载
    # root路径为下载路径，你也可以选择对应的文件夹位置下载
    # 不同数据集需要调用的接口可能不同，请阅读pytorch文档进行修改
    training_data = Dataset[name](
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # 不同数据集需要调用的接口可能不同，请阅读pytorch文档进行修改
    test_data = Dataset[name](
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)     


    examples = enumerate(test_data)
    batch_idx, (example_data, example_targets) = next(examples)

    channel = list(example_data.shape)[0]
    height  = list(example_data.shape)[1]
    width   = list(example_data.shape)[2]

    label_map = DATASET_CLASS[name]
    '''
    将数据张量以batch_size的大小进行划分
    关于batch_size：
    https://blog.csdn.net/qq_34886403/article/details/82558399/
    '''

    return (train_dataloader,test_dataloader,channel,height,width,test_data,label_map)
