"""
1.加载预训练vgg16模型
"""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


# 加载预训练的VGG16模型（使用默认权重）
vgg16 = models.vgg16(weights='DEFAULT')

# 获取VGG16模型的卷积和池化层（即特征提取部分）
vgg = vgg16.features

# 将所有卷积和池化层的参数设置为不可训练（即冻结参数）
for param in vgg.parameters():
    param.requires_grad_(False)



"""
2.加载猴子数据集
"""
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪为224x224大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，默认概率为0.5
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 图像标准化处理
])

val_data_transforms = transforms.Compose([
    transforms.Resize(256),  # 将图像短边缩放为256像素
    transforms.CenterCrop(224),  # 中心裁剪为224x224大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 图像标准化处理
])

train_data_dir = r"./monkey/training"  # 训练数据集的目录路径
train_data = ImageFolder(train_data_dir, transform=train_data_transforms)  # 加载训练数据集

train_loader = Data.DataLoader(
    train_data,
    batch_size=32,  # 每个批次的样本数量
    shuffle=True,  # 是否打乱数据顺序
)

val_data_dir = r'./monkey/validation'  # 验证数据集的目录路径
val_data = ImageFolder(val_data_dir, transform=val_data_transforms)  # 加载验证数据集
val_loader = Data.DataLoader(val_data, batch_size=32, shuffle=True)  # 创建验证数据加载器

print("训练集样本：", len(train_data.targets))  # 打印训练集样本数量
print("测试集样本：", len(val_data.targets))  # 打印验证集样本数量



# 展示图像
import matplotlib.pyplot as plt

# 获取一个批次的数据（32张图像）
images, labels = next(iter(train_loader))

# 将张量转换为numpy数组，并逆标准化处理
img = images[0].numpy().transpose((1, 2, 0))
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img = std * img + mean

# 显示图像
plt.imshow(img)
plt.title("Label: {}".format(labels[0]))
plt.show()



"""
3.微调VGG16模型
"""
class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel, self).__init__()

        # 定义模型结构
        self.vgg = vgg  # 特征提取层使用预训练的VGG模型

        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),  # 全连接层1，输入大小为特征提取层输出的展平向量大小
            nn.Tanh(),  # Tanh激活函数
            nn.Dropout(p=0.3),  # Dropout正则化层，丢弃率为0.3

            nn.Linear(512, 256),  # 全连接层2
            nn.Tanh(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 10),  # 全连接层3，输出大小为分类类别数
            nn.Softmax(dim=1)  # Softmax激活函数，用于多分类任务中输出概率分布
        )

    def forward(self, x):
        # 前向传播过程
        x = self.vgg(x)  # 提取特征
        x = x.view(x.size(0), -1)  # 将特征展平成向量
        output = self.classifier(x)  # 输入分类器，得到输出
        return output

    def initialize(self):
        # 参数初始化方法
        for m in self.modules():  # 遍历网络中的所有模块
            if isinstance(m, nn.Linear):  # 如果当前模块是全连接层
                tanh_gain = nn.init.calculate_gain('tanh')  # 计算Tanh激活函数的增益因子

                # 使用Xavier初始化方法对全连接层权重进行初始化
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定使用GPU或CPU运行模型
Myvggc = MyVggModel().to(device)  # 创建MyVggModel实例并将其移动到指定设备上
Myvggc.initialize()  # 参数初始化
optimizer = torch.optim.SGD(Myvggc.parameters(), lr=0.003)  # 创建一个随机梯度下降（SGD）优化器，使用 Myvggc 的可训练参数，并设置学习率为 0.003
loss_func = nn.CrossEntropyLoss()  # 定义交叉熵损失函数，用于衡量模型预测结果与真实标签的差异



"""
4.模型训练
"""
train_loss = []  # 训练集每轮的平均损失
train_c = []  # 训练集每轮的准确率

val_loss = []  # 验证集每轮的平均损失
val_c = []  # 验证集每轮的准确率

for epoch in range(20):  # 进行 50 轮训练

    train_loss_epoch = 0  # 初始化当前轮次的训练集总损失
    val_loss_epoch = 0  # 初始化当前轮次的验证集总损失
    train_corrects = 0  # 初始化当前轮次的训练集正确分类的样本数
    val_correct = 0  # 初始化当前轮次的验证集正确分类的样本数

    Myvggc.train()  # 将模型设为训练模式

    for step, (target, label) in enumerate(train_loader):  # 遍历训练数据集中每个批次的数据

        target, label = target.to(device), label.to(device)  # 将数据移动到 GPU 上

        output = Myvggc(target)  # 前向传播，得到模型的预测结果
        loss = loss_func(output, label)  # 计算预测结果与真实标签之间的交叉熵损失

        pre_lab = torch.argmax(output, 1)  # 预测结果中概率最大的类别即为预测标签

        optimizer.zero_grad()  # 清空优化器中之前的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新可训练参数

        train_loss_epoch += loss.item() * target.size(0)  # 累加当前批次的训练集损失
        train_corrects += torch.sum(pre_lab == label)  # 统计当前批次的正确分类样本数

    print('**'*10,'已完成', epoch+1,'轮')

    train_loss.append(train_loss_epoch / len(train_data.targets) )  # 将当前轮次的训练集平均损失添加到列表中
    train_c.append(train_corrects.cpu().numpy() / len(train_data.targets))  # 将当前轮次的训练集准确率添加到列表中

    Myvggc.eval()  # 将模型设为评估模式
    with torch.no_grad():

        for step, (target, label) in enumerate(val_loader):  # 遍历验证数据集中每个批次的数据

            target, label = target.to(device), label.to(device)  # 将数据移动到 GPU 上

            output = Myvggc(target)  # 前向传播，得到模型的预测结果
            loss = loss_func(output, label)  # 计算预测结果与真实标签之间的交叉熵损失

            pre_lab = torch.argmax(output, 1)  # 预测结果中概率最大的类别即为预测标签

            val_loss_epoch += loss.item() * target.size(0)  # 累加当前批次的验证集损失
            val_correct += torch.sum(pre_lab == label)  # 统计当前批次的正确分类样本数

        val_loss.append(val_loss_epoch / len(val_data.targets))  # 将当前轮次的验证集平均损失添加到列表中
        val_c.append(val_correct.cpu().numpy() / len(val_data.targets))  # 将当前轮次的验证集准确率添加到列表中

        print('Epoch: [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}%, Val Loss: {:.4f}, Val Accuracy: {:.4f}%'
              .format(epoch+1, 20, train_loss[-1], train_c[-1]*100, val_loss[-1], val_c[-1]*100))



# 保存模型及训练过程中的准确率和损失。
import scipy.io as scio
# 保存模型
torch.save(Myvggc,'vgg16_monky.pkl',_use_new_zipfile_serialization=False)
# 保存损失和准确率
scio.savemat('val_loss.mat',{'val_loss':val_loss})
scio.savemat('val_c.mat',{'val_c':val_c })
scio.savemat('train_loss.mat',{'train_loss':train_loss })
scio.savemat('train_c.mat',{'train_c':train_c })



"""
5.准确率损失可视化
"""
import numpy as  np
import matplotlib.pyplot as plt

val_loss = scio.loadmat('val_loss.mat')
val_acc = scio.loadmat('val_c.mat')
train_loss = scio.loadmat('train_loss.mat')
train_acc = scio.loadmat('train_c.mat')

val_loss = val_loss['val_loss']
val_acc = val_acc['val_c']
train_loss = train_loss['train_loss']
train_acc = train_acc['train_c']


plt.subplot(2,1,1)
plt.plot(val_loss[0], label='val_loss')
plt.plot(train_loss[0],label='train_loss')
plt.legend()
plt.ylabel('loss')

plt.subplot(2,1,2)
plt.plot(train_acc[0], label='train_acc')
plt.plot(val_acc[0], label='val_acc')
plt.legend()
plt.ylabel('acc')
plt.savefig('vgg16(80_epoch).png')
plt.show()

