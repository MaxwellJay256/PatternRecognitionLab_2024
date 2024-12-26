import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import warnings
import json
from tqdm import tqdm
from skimage import io
import skimage
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from visualize import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics import JaccardIndex  # 用于计算 IoU
import torch.nn.functional as F
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA is available, using GPU...")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available, using MPS...")
else:
    device = torch.device("cpu")
    print("CUDA and MPS are not available, using CPU...")

print('Current device: ' + str(torch.cuda.current_device()))
print('Device name: ' + str(torch.cuda.get_device_name(torch.cuda.current_device())))

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42
random.seed = seed
np.random.seed = seed
class BalloonDataset(Dataset):
    def __init__(self, annotations, dataset_dir, img_size=(128, 128), transform=None, cache_dir='cache'):
        self.annotations = annotations
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.transform = transform
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, f'{idx}.npz')
        if os.path.exists(cache_path):  # 如果缓存文件存在，则直接加载
            data = np.load(cache_path)
            image = torch.tensor(data['image'], dtype=torch.float32)
            mask = torch.tensor(data['mask'], dtype=torch.float32)
        else:  # 如果缓存文件不存在，则加载数据并保存到缓存文件
            tid = list(self.annotations.keys())[idx]
            a = self.annotations[tid]
            mask, image, _, _, _, _ = get_mask(a, self.dataset_dir)
            mask = resize(mask, self.img_size, mode='constant', preserve_range=True).astype(np.float32)
            image = resize(image, self.img_size, mode='constant', preserve_range=True).astype(np.float32) / 255.0
            if self.transform:
                image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            image = image.clone().detach().float()
            np.savez(cache_path, image=image.numpy(), mask=mask.numpy())
        return image, mask


def load_cached_data(cache_dir, idx):
    cache_path = os.path.join(cache_dir, f'{idx}.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        image = torch.from_numpy(data['image']).float()
        mask = torch.from_numpy(data['mask']).float()
        return image, mask
    else:
        raise FileNotFoundError(f'Cache file {cache_path} not found.')


def get_mask(a, dataset_dir):
    '''
    获取掩膜
    '''
    image_path = os.path.join(dataset_dir, a['filename'])
    image = io.imread(image_path)
    height, width = image.shape[:2]
    polygons = [r['shape_attributes'] for r in a['regions'].values()]
    mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

    for i, p in enumerate(polygons):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        # print(max(cc))
        rr = list(map(lambda x: height-1 if x > height-1 else x, rr))
        cc = list(map(lambda x: width-1 if x > width-1 else x, cc))
        # print("i:",i)
        mask[rr, cc, i] = 1

    mask, class_ids = mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    # boxes = extract_bboxes(mask)
    boxes = extract_bboxes(resize(mask, (128, 128), mode='constant', preserve_range=True))

    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]

    class_id = top_ids[0]
    # Pull masks of instances belonging to the same class.
    m = mask[:, :, np.where(class_ids == class_id)[0]]
    m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)

    return m, image, height, width, class_ids, boxes

### 加载数据集
print('加载训练数据集...')
annotations_path = "dataset/balloon/train_fake/via_region_data.json"
dataset_dir = 'dataset/balloon/train_fake'
annotations = json.load(open(annotations_path))
train_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = BalloonDataset(annotations, dataset_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 验证数据集
print('加载验证数据集...')
annotations_test_path = "dataset/balloon/val/via_region_data.json"
testset_dir = 'dataset/balloon/val'
annotations_test = json.load(open(annotations_test_path))
test_dataset = BalloonDataset(annotations_test, testset_dir, transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)
print('数据集加载完成。')


class DoubleConv(nn.Module):
    '''
    注意到每次 pool 操作之前，总是会卷积 2 次，将图片的通道数增加，
    所以我们可以将这个过程封装成一个类，避免重复写代码
    '''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(2, out_channels),  # 使用 GroupNorm 替代 BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(2, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, features: list):
        super(UNetModel, self).__init__()

        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        
        # 补全 UNetModel 网络结构
        self.out_channels = 1
        self.features = features  # 特征图通道数
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        
        # 添加下采样部分
        in_channels = self.IMG_CHANNELS  # 最开始输入为 3 通道
        for feature in self.features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 添加上采样部分
        for feature in reversed(self.features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(self.features[-1], self.features[-1]*2)  # 瓶颈层
        self.final_conv = nn.Conv2d(self.features[0], self.out_channels, kernel_size=1)  # 最后的输出层

    def forward(self, x):
        '''
        前向传播函数
        '''
        skip_connections = []  # 用于存储每次下采样后的特征图
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转列表

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


features = [8, 16, 32, 64, 128]
model = UNetModel(128, 128, 3, features).to(device)
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    best_model_wts = model.state_dict()
    best_iou = 0.0
    iou_metric = JaccardIndex(task='binary', num_classes=2).to(device)

    train_loss_list = []
    val_loss_list = []
    val_iou_list = []

    # 训练模型
    print('Start training...')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.bool()
            masks = masks.float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.bool()
            masks = masks.float()
            with torch.no_grad():
                outputs = model(images)
                bool = outputs[0] > 0.5  # 将输出的概率二值化
                bool = bool.float() * 255  # 将 0/1 转换为 0/255
                bool = bool.permute(1, 2, 0)  # 将通道放到最后

                image = masks[0].permute(1, 2, 0) * 255  # 将掩膜转换为图片
                # 掩膜二值化
                image = image.bool()

                # 计算 IoU
                iou = iou_metric(bool.unsqueeze(0), image.unsqueeze(0))
                val_iou += iou.item()

                # 计算 loss
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_list.append(val_loss)
        val_iou_list.append(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            best_model_wts = model.state_dict()

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

    model.load_state_dict(best_model_wts)

    # 绘制损失曲线
    plt.figure(figsize=(9, 5))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')
    # plt.show()

    # 绘制 IoU 曲线
    plt.figure(figsize=(9, 5))
    plt.plot(val_iou_list, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('IoU Curve')
    plt.savefig('iou_curve.png')
    # plt.show()

    return model


if __name__ == '__main__':
    num_epochs = 500
    model = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=num_epochs)
    # 将 model 保存到本地
    torch.save(model.state_dict(), 'models/best_model_' + str(num_epochs) + '.pth')
