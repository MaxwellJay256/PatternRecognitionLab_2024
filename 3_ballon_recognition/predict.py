import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型预测
import torch
from visualize import *
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from train import BalloonDataset, annotations_test, testset_dir, train_transform, UNetModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dataset = BalloonDataset(annotations_test, testset_dir, transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)
criterion = torch.nn.BCEWithLogitsLoss()
iou_metric = JaccardIndex(task='binary', num_classes=2).to(device)


def predict(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)  # 用模型预测图片中的掩膜
            preds.append(outputs.cpu().numpy())  # 将预测结果转移到 CPU 并添加到列表中
    return np.concatenate(preds, axis=0)  # 以 batch 为单位拼接结果


def plot_predictions(images, masks, preds):
    '''
    在一张图片里从左到右依次放置原图、真实掩膜、预测掩膜
    images: 图片
    masks: 真实掩膜 (ground truth)
    preds: 预测掩膜
    '''
    for i in range(len(images)):
        image = images[i].transpose(1, 2, 0)
        mask = masks[i]
        pred = preds[i]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image)
        axs[0].set_title("Image")
        axs[1].imshow(mask.squeeze())
        axs[1].set_title("Mask")
        axs[2].imshow(pred.squeeze())
        axs[2].set_title("Prediction")
        plt.savefig(f"./predictions/{i}.png")


# 加载模型
print('加载模型...')
model_name = 'best_model_500.pth'
features = [8, 16, 32, 64, 128]
model = UNetModel(128, 128, 3, features).to(device)
model.load_state_dict(torch.load("models/" + model_name, weights_only=True))

preds_test = predict(model, test_loader, device)
print('预测完成。')

test_images, test_masks = next(iter(test_loader))
plot_predictions(test_images.numpy(), test_masks.numpy(), preds_test)
print('预测结果已保存为图片。')


def calculate_metrics(model, dataloader, loss_criterion, iou_metric, device):
    '''
    计算平均 loss 和 ioU
    '''
    model.eval()
    sum_loss = []
    sum_iou = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.bool().float()  # 将掩膜转换为二进制
            with torch.no_grad():
                outputs = model(images)
                bool = outputs[0] > 0.5  # 将输出二值化
                bool = bool.float() * 255
                bool = bool.permute(1, 2, 0)

                image = masks[0].permute(1, 2, 0) * 255
                image = image.bool()

                loss = loss_criterion(outputs, masks)
                iou = iou_metric(bool.unsqueeze(0), image.unsqueeze(0))
                sum_loss.append(loss.item())
                sum_iou.append(iou.item())
    
    avg_loss = np.mean(sum_loss)
    avg_iou = np.mean(sum_iou)
    return avg_loss, avg_iou

print('计算平均 loss 和 IoU...')
test_loss, test_iou = calculate_metrics(model, test_loader, criterion, iou_metric, device)
print(f"Test loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")
