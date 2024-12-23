# coding: UTF-8
# 导入模块
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. 获取数据集
iris = load_iris()

# 2. 数据基本处理 - 数据集分割
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=2, test_size=0.2)
# random_state: 随机数种子，保证每次运行程序时，随机数相同
# test_size: 测试集占比，一般为 0.2 - 0.3

# 3. 特征工程
# 3.1 实例化一个转换器
transfer = StandardScaler()
# 3.2 调用 fit_transform 方法，对训练集和测试集的特征值进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 机器学习（训练模型）
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier(n_neighbors=5) # n_neighbors: 邻居个数
# 4.2 调用 fit 方法，进行训练
estimator.fit(x_train, y_train)

# 5. 模型评估
# 5.1 输出预测值
y_pre = estimator.predict(x_test)
print("预测类别为：\n", y_pre)
print("真实类别为：\n", y_test)
# 5.2 输出准确率
acc = estimator.score(x_test, y_test)
print("准确率为：\n", acc)
