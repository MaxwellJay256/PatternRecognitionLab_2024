# Pattern Recognition Lab 2024

HITSZ 2024 模式识别实验

## 0. Intro

- [iris_example.py](./0_intro/iris_example.py): Iris 数据集示例
- [knn_sort.py](./0_intro/knn_sort.py): KNN 最近邻算法分类示例

## 1. 人脸识别实验

- [fisher.py](./1_face_classification/fisher.py): 基于类内类间离散度的特征降维
- [fisher_mean.py](./1_face_classification/fisher_mean.py): 基于类内类间离散度的特征降维，并且去除均值

## 2. 字符识别实验

- 数据集：license_plate_number_dataset 中的 Char74K，包含除去字母 "O" 和 "I" 的所有英文大写字母和 0~9 的数字
- [LicensePlateRecognition_CNN_CPU/CC.py](/2_character_recognition/LicensePlateRecognition_CNN_CPU/CC.py): 基于卷积神经网络 CNN 的车牌字符识别
- [LicensePlateRecognition_DNN_CPU/CC_KLT.py](./2_character_recognition/LicensePlateRecognition_DNN_CPU/CC_KLT.py): 基于多层感知机 MLP 以及 K-L 变换的车牌字符识别，特征降维是关键

## 3. 气球识别实验

- [train.py](./3_ballon_recognition/train.py): 构建和训练 UNet 网络模型。
- [predict.py](./3_ballon_recognition/predict.py): 加载训练好的模型进行预测，统计损失和 IoU。
- 有必要使用 GPU 加速训练。
