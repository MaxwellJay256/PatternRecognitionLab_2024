# 车牌中的数字字母识别
# DNN: 深度神经网络，也称为多层感知机（Multi-Layer Perceptron, MLP）

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
# 模型构建
import torch.nn as nn # 神经网络模块
# 模型训练
import torch.optim as optim # 优化器
# 数据准备
import torchvision.transforms as transforms # 数据预处理
from torch.utils.data import DataLoader
from dataloader import CharacterDataset # 数据集

# --- 基本参数设置 ---
epochs = 100  # 训练轮数
batch_size = 1024  # 批处理大小
number_classes = 34  # 分割类别（含背景类）
epoch_best = 10  # 100 个 epoch 中较好的一组训练结果

img_size = 28 * 28  # 字符图片的大小，如果不降维，则特征维数就是这么多

def getX(dataset):
    '''
    获取数据集的样本矩阵 X
    '''
    X = []
    for i in range(1, len(dataset) + 1):
        img, label = dataset[i - 1]
        img = np.array(img, dtype=np.float32)
        img = img.flatten()
        X.append(img)
    mean = np.mean(X, axis=0)
    X = np.array(X) - mean  # 去均值
    return X, mean

def getEigenVector(X, e):
    '''
    获取数据集的特征向量
    '''
    Y = np.dot(np.transpose(X), X) / X.shape[0]  # 样本矩阵降维：m*wh * wh*m -> m*m
    eigenvalue, feature_vector = np.linalg.eig(Y)  # 计算特征值和特征向量
    feature_vector = np.transpose(feature_vector)  # 将特征向量转置
    eigen = []
    for i in range(0, eigenvalue.shape[0]):
        eigen.append([eigenvalue[i], feature_vector[i]])
    eigen = sorted(eigen, key=lambda eigen: eigen[0], reverse=True)  # 根据特征值大小从大到小排序
    eigen_num = 0  # 根据重构精度确定特征数目
    eigenvalue_sum = 0
    for i in range(0, eigenvalue.shape[0]):
        eigenvalue_sum += eigen[i][0]
        if eigenvalue_sum / np.sum(eigenvalue) >= e:
            eigen_num = i + 1
            break
    
    eigen_vector = []
    for i in range(0, eigen_num):
        vec = eigen[i][1]
        eigen_vector.append(vec)
    
    return eigen_num, np.array(eigen_vector)

def transform_data(dataset, mean, eigen_vector):
    '''
    对数据集进行 K-L 变换，特征降维
    '''
    data_transformed = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = np.array(img).flatten()
        img = img - mean
        img_transformed = np.dot(eigen_vector, img)
        data_transformed.append([img_transformed, label])
    return data_transformed

# ------------------------------------------------------------ 定义网络结构 ------------------------------------------------------------
print('构建模型……')

class Classification_KLT(nn.Module):
    def __init__(self, num, vec):
        super(Classification_KLT, self).__init__()
        # 定义网络结构
        self.flatten = nn.Flatten()
        self.vec = vec
        self.output = nn.Sequential(
            nn.Linear(num, 512),  # 全连接层，128 个神经元
            nn.ReLU(True),  # 激活函数
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),  # Dropout 层，防止过拟合
            nn.Linear(128, number_classes),  # 全连接层，34 个神经元
        )
    # 定义前向传播函数
    def forward(self, x):
        # print("x.shape:", x.shape)
        # print("vec.shape:", self.vec.shape)
        x = self.flatten(x)  # 将输入展平为一维
        output = self.output(x)
        return output

# ------------------------------------------------------------ 网络训练 ------------------------------------------------------------
# 载入并更新网络、优化器权重参数（在已有模型的基础上继续训练）
# weights_net = torch.load('./models_KLT/CC_final.pth', map_location='cpu')
# CC_KLT.load_state_dict(weights_net)
# weights_optimizer = torch.load('./models_KLT/CC_optimizer_final.pth', map_location='cpu')
# optimizer.load_state_dict(weights_optimizer)

# 训练数据准备
print('载入数据……')
path = './dataset/train.txt'
transform = transforms.Compose([transforms.ToTensor(), ])
characters_train = CharacterDataset(path, transform=transform)
data_loader_train = DataLoader(characters_train, batch_size=batch_size, shuffle=True)

# 测试数据准备
path = './dataset/test.txt'
transform = transforms.Compose([transforms.ToTensor(), ])
characters_test = CharacterDataset(path, transform=transform)
data_loader_test = DataLoader(characters_test, batch_size=batch_size, shuffle=False)

# 对数据集先进行 K-L 变换
print('开始 K-L 变换……')
X_train, mean_train = getX(characters_train)
print("训练集均值计算完成。")
X_test, mean_test = getX(characters_test)
print('测试集均值计算完成。')

refactor_accuracy = 0.95  # 重构精度
eigen_num, eigen_vector = getEigenVector(X_train, refactor_accuracy)
print(f"K-L 变换后的特征维数：{eigen_num}")

train_data_transformed = transform_data(characters_train, mean_train, eigen_vector)
test_data_transformed = transform_data(characters_test, mean_test, eigen_vector)

print('载入变换后的数据……')

data_loader_train = DataLoader(train_data_transformed, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(test_data_transformed, batch_size=batch_size, shuffle=False)

CC_KLT = Classification_KLT(eigen_num, torch.tensor(eigen_vector, dtype=torch.float32))
lossFun = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(CC_KLT.parameters())

# 模型训练
loss_epochs_train = []  # 训练过程中每个 epoch 的平均损失保存在此列表中，用于显示
loss_epochs_test = []
epochs_x = []  # 显示用的横坐标
print('开始训练……')
for epoch in range(epochs):
    CC_KLT.train()  # 设置为训练模式
    loss_train = []
    for index, (images, labels) in enumerate(data_loader_train):
        CC_output   = CC_KLT(images)
        loss        = lossFun(CC_output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data.item())
        # print('epoch:{}, batch:{}, loss:{:.6f}'.format(epoch+1, index+1, loss.data.item()))
    if (epoch + 1) % 10 == 0:  # 每训练 10 个 epoch 保存一次模型
        os.makedirs('./models_KLT', exist_ok=True)
        torch.save(CC_KLT.state_dict(), './models_KLT/CC_'+str(epoch+1)+'.pth')
        torch.save(optimizer.state_dict(), './models_KLT/CC_optimizer_'+str(epoch+1)+'.pth')

    # 模型测试（每个 epoch 使用测试集进行一次验证）
    CC_KLT.eval()
    loss_test = []
    print('开始验证……')
    for index, (images, labels) in enumerate(data_loader_test):
        CC_output   = CC_KLT(images)
        loss        = lossFun(CC_output, labels)
        loss_test.append(loss.data.item())
    print('平均损失：', np.mean(loss_test))

    # 记录损失
    loss_epochs_train.append(np.mean(loss_train))
    loss_epochs_test.append(np.mean(loss_test))
    epochs_x.append(epoch + 1)

# 最终模型保存
torch.save(CC_KLT.state_dict(), './models_KLT/CC_final.pth')
torch.save(optimizer.state_dict(), './models_KLT/CC_optimizer_final.pth')

# 损失可视化（需要安装 matplotlib 库，安装指令：conda install matplotlib）
import matplotlib.pyplot as plt
plt.figure('loss')
plt.plot(epochs_x, loss_epochs_train, label='Training loss')
plt.plot(epochs_x, loss_epochs_test, label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(frameon=False)
plt.savefig('./loss_KLT.png')
plt.show()

epoch_best = int(loss_epochs_test.index(min(loss_epochs_test)) / 10 + 0.6) * 10
if epoch_best < 10:
    epoch_best = 10
print('epoch_best:', epoch_best)

# ------------------------------------------------------------ 网络测试 ------------------------------------------------------------
# 识别测试
import os
import shutil
from PIL import Image

# 模型实例化
CC_KLT = Classification_KLT(eigen_num, torch.tensor(eigen_vector, dtype=torch.float32))
CC_KLT.eval() # 设置为评估模式
# 载入并更新网络权重参数
weights_net = torch.load('./models_KLT/CC_' + str(epoch_best) + '.pth', map_location='cpu', weights_only=True)
CC_KLT.load_state_dict(weights_net)

softmax = nn.Sequential(nn.Softmax(dim=1))
# 构建结果索引
index_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'J', 19:'K',
    20:'L', 21:'M', 22:'N', 23:'P', 24:'Q', 25:'R', 26:'S', 27:'T', 28:'U', 29:'V',
    30:'W', 31:'X', 32:'Y', 33:'Z'
}

# 测试集分类测试
os.makedirs('./output_KLT', exist_ok=True)
for name in os.listdir('./output'):
    path = './output_KLT/' + name
    if os.path.exists(path):
        os.remove(path)

test_path = './dataset/test.txt'
number = 0
error_num = 0
rate = 0

with open(test_path, 'r', encoding='utf-8') as test:
    for line in test.readlines():
        line = line.rstrip()
        words = line.split()
        label = words[-1]
        path = line[:-(label.__len__()+1)]
        label = int(label)
        image_name = path.split('/')[-1]

        image = Image.open(path).convert('L')
        transform = transforms.Compose([transforms.ToTensor(), ])
        image = transform(image).view(1, 1, 28, 28)
        
        # 对输入数据进行K-L变换
        image = image.numpy().flatten()
        image = image - mean_train  # 使用训练集的均值
        image = np.dot(eigen_vector, image)
        image = torch.tensor(image, dtype=torch.float32).view(1, -1)  # 转换为torch tensor并调整维度
        
        output = softmax(CC_KLT(image))
        index = output.topk(1)[1].numpy()[0][0]
        if label != index:
            error_num += 1
            print('Wrong! label: {}, prediction: {}, image_name: {}'.format(index_dict[label], index_dict[index], image_name))
            target_path = './output_KLT/' + index_dict[label] + '_' + index_dict[index] + '_' +image_name
            shutil.copyfile(path, target_path)
        number += 1
print('测试图像数量: {}, 误分类数量: {}, 分类准确率: {:.2f}%'.format(number, error_num, (1-error_num/number)*100))
