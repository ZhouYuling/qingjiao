"""
任务一：数据处理
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义数据预处理的transform
transform = transforms.Compose([
    # 随机水平翻转图像
    transforms.RandomHorizontalFlip(),
    # 转换为tensor类型
    transforms.ToTensor(),
    # 归一化到[-1, 1]之间
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载CIFAR10训练集和测试集
train_set = torchvision.datasets.CIFAR10('./datasets', train=True,
                                         download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10('./datasets', train=False,
                                        download=True, transform=transform)

# 构建训练集和测试集的dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                           shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                          shuffle=False, num_workers=4)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

# 定义显示图像的函数
def imshow(img):
    img = img / 2 + 0.5
    # 将图像数据从tensor格式转换为numpy数组，并改变通道维度的顺序
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    # 显示图像
    plt.show()

# 创建一个迭代器用于遍历训练集数据
image_iter = iter(train_loader)
# 获取下一个批次的图像和对应的标签（这里用不到标签，所以用_表示）
images, _ = next(image_iter)
# 调用imshow函数显示前4张图像
imshow(torchvision.utils.make_grid(images[:4]))


"""
任务2：残差块
"""
class BasicBlock(nn.Module):
    """
    expansion：扩展因子，指定捷径连接中使用的1x1卷积层的输出通道数与特征提取部分输出通道数之间的比例，默认为1。
    对于浅层网络，我们使用基本的Block
    基础块没有维度压缩，所以expansion=1
    特征提取部分：卷积核大小为3×3，卷积步长为1，填充为1，经过卷积后的特征图大小不变
    捷径连接部分：如果卷积前后特征图数量（通道数）不相同，则利用shortcut改变捷径连接部分的输出通道数
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 定义基本块的特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积层，输入通道数为in_channels，输出通道数为out_channels，
            # 卷积核大小为3x3，步长为stride，padding为1，不使用偏置参数
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # 对输出通道数为out_channels的特征图进行批归一化处理
            nn.BatchNorm2d(out_channels),
            # 使用ReLU激活函数
            nn.ReLU(inplace=True),
            # 第二个卷积层，输入通道数为out_channels，输出通道数为out_channels，
            # 卷积核大小为3x3，步长为1，padding为1，不使用偏置参数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # 对输出通道数为out_channels的特征图进行批归一化处理
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        # 如果输入输出维度不等，则使用1x1卷积层来改变维度
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积层来改变通道数，输入通道数为in_channels，输出通道数为self.expansion * out_channels，
                # 步长为stride，不使用偏置参数
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                # 对输出通道数为self.expansion * out_channels的特征图进行批归一化处理
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        # 将输入数据传入特征提取部分
        out = self.features(x)
        # 如果输入输出维度不等，则使用shortcut改变维度
        out += self.shortcut(x)
        # 使用ReLU激活函数
        out = torch.relu(out)
        # 返回输出数据
        return out



# 测试
# Bottleneck对象bottleneck，输入通道数为64，压缩后的维数为128
basic_block = BasicBlock(64, 128)
# 打印结构
print(basic_block)
# 创建一个随机张量x，大小为(2, 256, 32, 32)
x = torch.randn(2, 64, 32, 32)
# 使用bottleneck对x进行前向传播得到输出结果y
y = basic_block(x)
print(y.shape)



class Bottleneck(nn.Module):
    """
    对于深层网络，我们使用BottleNeck，论文中提出其拥有近似的计算复杂度，但能节省很多资源
    zip_channels: 压缩后的维数，最后输出的维数是 expansion * zip_channels。其作用是降低特征图个数（通道数），减少计算量。
    特征提取部分：特征图大小依然保持不变。
    捷径连接部分：如果卷积前后特征图数量（通道数）不相同，则利用shortcut改变捷径连接部分的输出通道数
    """
    expansion = 4  # 扩展因子，用于计算输出通道数

    def __init__(self, in_channels, zip_channels, stride=1):
        super(Bottleneck, self).__init__()
        out_channels = self.expansion * zip_channels  # 计算输出通道数

        # 定义BottleNeck块的特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积层，输入通道数为in_channels，输出通道数为zip_channels，
            # 卷积核大小为1x1，不使用偏置参数
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            # 对输出通道数为zip_channels的特征图进行批归一化处理
            nn.BatchNorm2d(zip_channels),
            # 使用ReLU激活函数
            nn.ReLU(inplace=True),

            # 第二个卷积层，输入通道数为zip_channels，输出通道数为zip_channels，
            # 卷积核大小为3x3，步长为stride，padding为1，不使用偏置参数
            nn.Conv2d(zip_channels, zip_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # 对输出通道数为zip_channels的特征图进行批归一化处理
            nn.BatchNorm2d(zip_channels),
            # 使用ReLU激活函数
            nn.ReLU(inplace=True),

            # 第三个卷积层，输入通道数为zip_channels，输出通道数为out_channels，
            # 卷积核大小为1x1，不使用偏置参数
            nn.Conv2d(zip_channels, out_channels, kernel_size=1, bias=False),
            # 对输出通道数为out_channels的特征图进行批归一化处理
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()  # identity mapping的shortcut

        # 如果输入输出通道数不一致或stride不为1，需要进行维度变换
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积层来改变通道数，输入通道数为in_channels，输出通道数为out_channels，
                # 步长为1，不使用偏置参数
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # 对输出通道数为out_channels的特征图进行批归一化处理
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 将输入数据传入特征提取部分
        out = self.features(x)
        # 加上identity mapping的shortcut
        out += self.shortcut(x)
        # 使用ReLU激活函数
        out = torch.relu(out)
        # 返回输出数据
        return out



# 测试
# Bottleneck对象bottleneck，输入通道数为256，压缩后的维数为128
bottleneck = Bottleneck(256, 128)
# 然后使用print打印输出了bottleneck的信息，包括其结构和参数
print(bottleneck)
# 创建一个随机张量x，大小为(2, 256, 32, 32)
x = torch.randn(2, 256, 32, 32)
# 使用bottleneck对x进行前向传播得到输出结果y
y = bottleneck(x)
print(y.shape)



"""
3.搭建 ResNet 网络
"""
class ResNet(nn.Module):
    """
    不同的ResNet架构都是统一的一层特征提取、四层残差，不同点在于每层残差的深度。
    对于cifar10，feature map size的变化如下：
    (32, 32, 3) -> [Conv2d] -> (32, 32, 64) -> [Res1] -> (32, 32, 64) -> [Res2]
 -> (16, 16, 128) -> [Res3] -> (8, 8, 256) ->[Res4] -> (4, 4, 512) -> [AvgPool]
 -> (1, 1, 512) -> [Reshape] -> (512) -> [Linear] -> (10)
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 定义特征提取部分，包括一个输入通道数为3，输出通道数为64的卷积层，卷积核大小为3x3，步长为1，padding为1，不使用偏置参数；一个输出通道数为64的批归一化层；一个ReLU激活函数。
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 定义四层残差结构
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 定义全局平均池化层，池化核大小为4x4
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        # 定义分类器，包括一个输入维度为512 * block.expansion，输出维度为num_classes的线性层
        self.classifer = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个block要进行降采样
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # 创建一个残差块并添加到layers列表中
            layers.append(block(self.in_channels, out_channels, stride))
            # 如果是Bottleneck Block的话需要对每层输入的维度进行压缩，压缩后再增加维数
            # 所以每层的输入维数也要跟着变
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 将输入数据传入特征提取部分
        out = self.features(x)
        # 将特征图传入四层残差结构中
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 进行全局平均池化
        out = self.avg_pool(out)
        # 将输出特征进行reshape
        out = out.view(out.size(0), -1)
        # 通过分类器进行分类
        out = self.classifer(out)
        return out



def ResNet18():
    # 返回一个使用BasicBlock和[2,2,2,2]作为参数的ResNet实例
    return ResNet(BasicBlock, [2,2,2,2])
def ResNet34():
    # 返回一个使用BasicBlock和[3,4,6,3]作为参数的ResNet实例
    return ResNet(BasicBlock, [3,4,6,3])
def ResNet50():
    # 返回一个使用Bottleneck和[3,4,6,3]作为参数的ResNet实例
    return ResNet(Bottleneck, [3,4,6,3])
def ResNet101():
    # 返回一个使用Bottleneck和[3,4,23,3]作为参数的ResNet实例
    return ResNet(Bottleneck, [3,4,23,3])
def ResNet152():
    # 返回一个使用Bottleneck和[3,8,36,3]作为参数的ResNet实例
    return ResNet(Bottleneck, [3,8,36,3])



# 根据是否支持cuda选择设备类型，如果支持则使用cuda，否则使用cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 创建一个ResNet34实例，并将其移动到指定的设备上
net = ResNet34().to(device)
# 打印网络结构
print(net)
# 如果设备类型为cuda
if device == 'cuda':
    # 则使用DataParallel进行模型并行处理
    net = nn.DataParallel(net)
    # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
    torch.backends.cudnn.benchmark = True


"""
4.模型训练
"""
# 设置优化器与损失函数。
lr = 1e-1
# 学习率
momentum = 0.9
# 动量
weight_decay = 5e-4
# 权重衰减

criterion = nn.CrossEntropyLoss()
# 损失函数为交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# 优化器为随机梯度下降，学习率为lr，动量为momentum，权重衰减为weight_decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.1, patience=3, verbose=True)
# 学习率调度器采用的是ReduceLROnPlateau方法，当监控指标不再变化时，将学习率按照factor进行缩放
# threshold表示学习率的变化量阈值，patience表示等待的epoch数，verbose表示是否打印日志信息



# 定义训练函数。
def train(epoch):
    print('\nEpoch: %d' % (epoch))
    # 打印当前的epoch数
    net.train()
    # 将模型设置为训练模式，启用batch normalization和dropout
    train_loss = 0
    # 记录训练损失
    correct = 0
    # 记录正确的预测数量
    total = 0
    # 记录总样本数量

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 遍历训练数据集中的每个batch
        inputs, targets = inputs.to(device), targets.to(device)
        # 将输入数据和目标值移动到指定的设备上

        optimizer.zero_grad()
        # 梯度清零，避免累计梯度影响训练
        outputs = net(inputs)
        # 通过网络进行前向传播，得到预测结果
        loss = criterion(outputs, targets)
        # 计算损失值
        loss.backward()
        # 反向传播，计算梯度
        optimizer.step()
        # 更新模型参数

        train_loss += loss.item()
        # 累计训练损失值
        _, predicted = outputs.max(1)
        # 获取预测结果中的最大值及其对应的索引
        total += targets.size(0)
        # 累计样本数量
        correct += predicted.eq(targets).sum().item()
        # 累计正确的预测数量
        if batch_idx % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.6f | Acc: %.3f%% (%d/%d)' %
                  (epoch + 1, batch_idx + 1, train_loss, 100.*correct/total, correct, total))
            # 每100个mini-batches打印一次训练损失和准确率

    return train_loss



# 加载模型。
# 设置是否加载已有的模型参数，默认为False，即不加载
load_model = False
# 如果load_model为True，则加载已有的模型参数
if load_model:
    # 加载保存的模型参数，文件路径为'./checkpoint/res18.ckpt'
    checkpoint = torch.load('./checkpoint/res18.ckpt')
    # 将加载的模型参数设置到网络模型中
    net.load_state_dict(checkpoint['net'])
    # 获取加载的模型参数所对应的epoch数
    start_epoch = checkpoint['epoch']
# 如果load_model为False，则不加载已有的模型参数
else:
    # 将起始epoch数设置为0，表示从头开始训练
    start_epoch = 0
# 打印起始epoch数
print('start_epoch: %s' % start_epoch)



# 模型训练。
# 循环遍历从起始epoch到10（不包括10）的范围
for epoch in range(start_epoch, 10):
    # 调用train函数进行训练，并获取损失值
    loss = train(epoch)
    # 打印损失值，保留小数点后6位
    print('Total loss: %.6f' % loss)
    # 更新起始epoch为当前epoch
    start_epoch = epoch
    # 调用scheduler.step()方法，根据损失值更新学习率
    scheduler.step(loss)



# 保存模型。
# 设置是否保存模型参数，默认为True，即保存模型参数
save_model = True

# 如果save_model为True，则保存模型参数
if save_model:
    # 创建一个字典state，包含网络模型参数net和当前epoch数epoch
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }

    # 使用os.makedirs()方法创建checkpoint文件夹，如果该文件夹已存在，则不进行任何操作
    os.makedirs('checkpoint', exist_ok=True)

    # 使用torch.save()函数将state字典保存到'./checkpoint/res18.ckpt'文件中
    torch.save(state, './checkpoint/res18.ckpt')



"""
5.模型验证
"""
# 创建一个迭代器dataiter，用于遍历test_loader中的数据
dataiter = iter(test_loader)

# 从迭代器中获取下一批数据，images为图像数据，labels为标签数据
images, labels = next(dataiter)
# 只选择前4个图像和对应的标签
images = images[:4]
labels = labels[:4]

# 使用torchvision.utils.make_grid()方法将图像数据转换为可视化的网格形式，并用imshow()函数显示图像
imshow(torchvision.utils.make_grid(images))

# 打印出图像对应的真实标签，使用空格分隔表示
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 将图像数据传入网络模型进行预测，得到预测结果
outputs = net(images.to(device))

# 使用torch.max()函数找到每行中最大值及其索引，即找到预测结果中概率最高的类别
_, predicted = torch.max(outputs.cpu(), 1)

# 打印出预测结果，使用空格分隔表示
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 初始化正确预测的样本数和总样本数为0
correct = 0
total = 0

# 关闭梯度计算，因为在测试阶段不需要进行梯度更新
with torch.no_grad():
    # 遍历测试数据集
    for data in test_loader:
        # 获取图像数据和标签数据
        images, labels = data
        # 将图像数据和标签数据移动到设备上（如GPU）
        images, labels = images.to(device), labels.to(device)
        # 将图像数据传入网络模型进行预测，得到预测结果
        outputs = net(images)
        # 使用torch.max()函数找到每行中最大值及其索引，即找到预测结果中概率最高的类别
        _, predicted = torch.max(outputs.data, 1)
        # 更新总样本数
        total += labels.size(0)
        # 计算正确预测的样本数
        correct += (predicted == labels).sum().item()

# 打印出网络模型在测试数据集上的准确率
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 初始化每个类别的正确预测样本数和总样本数为0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# 关闭梯度计算
with torch.no_grad():
    # 遍历测试数据集
    for data in test_loader:
        # 获取图像数据和标签数据
        images, labels = data
        # 将图像数据和标签数据移动到设备上（如GPU）
        images, labels = images.to(device), labels.to(device)
        # 将图像数据传入网络模型进行预测，得到预测结果
        outputs = net(images)
        # 使用torch.max()函数找到每行中最大值及其索引，即找到预测结果中概率最高的类别
        _, predicted = torch.max(outputs, 1)
        # 将预测结果与标签进行比较，得到一个布尔值张量
        c = (predicted == labels).squeeze()
        # 获取当前样本的真实标签
        for i in range(4):
            label = labels[i]
            # 如果预测正确，则对应类别的正确预测样本数加1
            class_correct[label] += c[i].item()
            # 对应类别的总样本数加1
            class_total[label] += 1

# 遍历每个类别
for i in range(10):
    # 打印出每个类别的准确率
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
