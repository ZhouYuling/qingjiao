import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

# 定义预处理的转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像
])

# 加载训练集和测试集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)



# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        x = x.view(-1, 32 * 7 * 7)

        x = self.fc(x)

        return x

# 创建CNN模型实例
model = CNN()



# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降优化器

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有可用的GPU加速
model.to(device)  # 将模型移动到对应的设备上进行训练



num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0  # 用于追踪每个epoch的累计损失
    correct = 0  # 用于追踪每个epoch的正确分类数量
    total = 0  # 用于追踪每个epoch的总样本数量

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()  # 梯度清零

        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(trainloader), running_loss/100))
            running_loss = 0.0

    accuracy = 100 * correct / total
    print('Epoch [{}/{}], Accuracy on the training set: {:.2f}%'.format(epoch+1, num_epochs, accuracy))

print('Finished Training')

# 在测试集上评估模型
model.eval()  # 将模型设置为评估模式，不进行梯度计算
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy on the test set: {:.2f}%'.format(accuracy))
