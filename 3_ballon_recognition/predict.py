# 模型预测
import torch
from visualize import *
from torch.utils.data import DataLoader, Dataset
from train import BalloonDataset, annotations_test, testset_dir, train_transform, UNetModel,train_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dataset = BalloonDataset(annotations_test, testset_dir, transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)

#
model = UNetModel(128,128,3).to(device)
model.load_state_dict(torch.load("./best_model_300.pth"))

preds_test = predict(model, test_loader, device)
##### need to be done ######
# 补全predict，保存预测掩模
#
test_images, test_masks = next(iter(test_loader))
plot_predictions(test_images.numpy(), test_masks.numpy(), preds_test)
##### need to be done ######
# 附加： 补全plot_predictions
# 计算平均loss和ioU，并保存预测研磨图片