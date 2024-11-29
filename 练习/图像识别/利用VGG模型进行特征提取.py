"""
9.2 利用VGG模型进行特征提取
VGG（Visual Geometry Group）是一种深度卷积神经网络模型，其主要用途是图像分类和特征提取。在VGG模型中，特征提取是通过卷积层来实现的。

利用VGG进行特征提取的一般步骤如下：

准备数据集：目标图像数据集。
加载预训练模型：VGG模型是在大规模图像数据集上进行预训练的，因此可以使用已经训练好的VGG模型来提取特征。
图像预处理：在输入图像之前，需要进行预处理，如调整图像大小、归一化像素等。这些步骤可以确保输入图像与预训练模型的要求相匹配。
特征提取：在VGG模型中，通常会提取最后一个全连接层（也称为特征向量）之前的所有层的输出作为特征。这些输出可以看作是对输入图像不同抽象级别的表示。
"""


"""
1.图形的大小和格式
"""
from PIL import Image

# 打开图像文件
image1 = Image.open('lenna.jpg')
image2 = Image.open('apple.jpg')

width1, height1 = image1.size
width2, height2 = image2.size

print(f"lenna图像宽度：{width1}, lenna图像高度：{height1}")
print(f"apple图像宽度：{width2}, apple图像高度：{height2}")
# 获取图像的模式（颜色通道顺序）
print("lenna's channel:", image1.mode)
print("apple's channel:", image2.mode)



from PIL import Image

# 读取带有透明度通道的图像
image = Image.open("lenna.jpg")

# 将图像转换为不带透明度通道的 "RGB" 模式
image = image.convert("RGB")

# 现在，image_without_alpha 就是去除了透明度通道的图像
image.mode



import cv2

# 读取图像文件
image1 = cv2.imread('lenna.jpg')
image2 = cv2.imread('apple.jpg')

# 获取图像的颜色通道
print(image1.shape)  # 输出 (height, width, channels)
print(image2.shape)


"""
2.三通道图像
"""
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import matplotlib.pyplot as plt  # 导入Matplotlib库
from PIL import Image  # 导入Pillow库中的Image类
from torchvision import models, transforms  # 导入torchvision库中的预训练模型和图像变换函数

# 定义图像变换操作，包括Resize、ToTensor和Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像大小调整为224x224
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化张量
])
# 加载图像并进行图像变换
image = transform(Image.open('apple.jpg')).unsqueeze(0)




# 加载预训练模型VGG19
model = models.vgg19(weights='DEFAULT')

# 获取VGG19模型的前5个卷积层，并对图像进行特征提取
features = model.features[:5](image)

# 将特征张量转换为NumPy数组，并获取第一个特征张量（即第1个卷积层的输出）
feature_map = features[0].detach().numpy()



# 将feature_map前16个特征图以4行4列的形式显示
fig, ax = plt.subplots(8, 8, figsize=(10, 10))
for i in range(8):
    for j in range(8):
        ax[i, j].imshow(feature_map[i * 4 + j], cmap='gray')
        ax[i, j].axis('off')
plt.show()



"""
3.四通道图像
"""
import torch  # 导入PyTorch库
from torchvision import models, transforms  # 从torchvision库中导入模型和数据转换函数
from PIL import Image  # 导入PIL库中的图像处理函数
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
import numpy as np  # 导入NumPy库用于数组操作
import scipy.misc  # 导入scipy库中的图像处理函数
import cv2  # 导入OpenCV库用于图像处理
import imageio  # 导入imageio库用于图像读写操作

# 定义图像文件路径
image_dir = 'lenna.jpg'

# 使用PIL库打开并转换图像为RGB格式
image = Image.open(image_dir).convert('RGB')

# 定义图像数据预处理操作
transform = transforms.Compose([
    transforms.Resize(1024),  # 调整图像大小
    transforms.CenterCrop(672),  # 中心裁剪
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化操作
])

# 应用预处理操作到图像上
image = transform(image)
# 将图像数据添加一个维度，以符合模型输入要求
image = image.unsqueeze(0)

model = models.vgg16(weights='DEFAULT')

feature_extractor = model.features
feature_extractor


#可视化特征图
feature_map = feature_extractor[:2](image).squeeze(0)#squeeze(0)实现tensor降维，开始将数据转化为图像格式显示
feature_map = feature_map.detach().numpy()#进行卷积运算后转化为numpy格式
feature_map_num = feature_map.shape[0]#特征图数量等于该层卷积运算后的特征图维度
print('feature_map_num', feature_map_num)



# 将feature_map前16个特征图以4行4列的形式显示
fig, ax = plt.subplots(8, 8, figsize=(10, 10))
for i in range(8):
    for j in range(8):
        ax[i, j].imshow(feature_map[i * 8 + j], cmap='gray')
        ax[i, j].axis('off')
plt.show()
del image



"""
4.create_feature_extractor
"""
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库
from PIL import Image  # 导入Pillow库中的Image类
import torchvision.transforms as transforms  # 导入torchvision库中的图像变换函数
from matplotlib import pyplot as plt  # 导入Matplotlib库
from torchvision import models  # 从torchvision.models.feature_extraction模块导入create_feature_extractor函数
from torchvision.models import feature_extraction

# 定义图像变换操作
transform = transforms.Compose([
    transforms.Resize(256),  # 调整图像大小为256x256
    transforms.CenterCrop(224),  # 对图像进行中心裁剪，尺寸为224x224
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化张量
])

# 使用预训练的ResNet18模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 创建特征提取器
feature_extractor = feature_extraction.create_feature_extractor(model, return_nodes={"conv1":"output"})
feature_extractor



nodes, _ = feature_extraction.get_graph_node_names(model)
nodes



for name in model.named_children():
    print(name[0])



# 读取图像
image = Image.open("apple.jpg")

# 对图像进行图像变换并增加一个维度作为批次维度
image = transform(image).unsqueeze(0)

# 提取特征
out = feature_extractor(image)
out



# 可视化特征图（未分通道）
plt.imshow(out["output"][0].transpose(0, 1).sum(1).detach().numpy())



# 可视化特征图（未分通道）
feature_map = out["output"].transpose(0, 1).sum(1).detach().numpy()

# 将特征图以4行4列的形式显示
fig, ax = plt.subplots(8, 8, figsize=(10, 10))
for i in range(8):
    for j in range(8):
        ax[i, j].imshow(feature_map[i * 8 + j], cmap='gray')  # 在子图中显示特征图
        ax[i, j].axis('off')  # 关闭子图的坐标轴
plt.show()  # 显示画布


