# coding:utf-8
# 导入模块
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# 1. 获取数据集
iris = load_iris()
print(iris)

# 2. 描述数据集属性
print("数据集集中特征值是：\n", iris.data)
print("数据集中目标值是：\n", iris.target)
print("数据集中特征值的名字是：\n", iris.feature_names)
print("数据集中目标值的名字是：\n", iris.target_names)
print("数据集的描述是：\n", iris.DESCR)

# 3. 数据可视化
# 3.1 数据集类型转换，把数据用 DataFrame 类型表示
iris_data = pd.DataFrame(data=iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_data["target"] = iris.target
print(iris_data)

# 3.2 画图
def iris_plot(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data, hue="target", fit_reg=False)
    plt.title("Iris Data Visualization")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

iris_plot(iris_data, "Sepal_Length", "Sepal_Width")
# 尝试可视化其他两个特征
iris_plot(iris_data, "Petal_Length", "Petal_Width")
