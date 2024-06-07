# 导入必要的库
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
# data = pd.read_excel('../data/ppplus_no_score.xlsx')
# X = data.iloc[:, 2:28].values
# y = data.iloc[:, 28].values
# features = data.columns[2:28].values

# data = pd.read_excel('../data/data2_final.xlsx')
# X = data.iloc[:, 2:29].values
# Y = data.iloc[:, 29].values
# features = data.columns[2:29].values
# print(features[5])
# print(features[20])
#
# # 初始化四个列表用于存储数据
# X1, Y1, X2, Y2 = [], [], [], []
#
# # 定义要移除的列索引
# columns_to_remove = [5, 20]
#
# # 遍历数据
# for i, x in enumerate(X):
#     # 创建临时变量存储移除指定列后的特征向量
#     temp_x = np.delete(x, columns_to_remove)
#     if (x[20] == 5 or x[20] == 6) and x[5] == 18:
#         X1.append(temp_x)
#         Y1.append(Y[i])  # 直接使用索引i获取Y的值，避免多次查找
#     else:
#         X2.append(temp_x)
#         Y2.append(Y[i])
# X = np.array(X2)
# y = np.array(Y2)
#
# features = np.delete(features, columns_to_remove)

# data2_final加载数据
data = pd.read_excel('../data/data2_final.xlsx', sheet_name='sheet2')
X = data.iloc[:, :].values
y = data.iloc[:, 58].values

# 定义要移除的列索引
columns_to_remove = [0, 1, 2, 3, 4, 5, 7, 5, 3, 54, 55, 56, 57, 58, 59]

X = np.delete(X, columns_to_remove, axis=1)
features = data.columns[:].values
features = np.delete(features, columns_to_remove)

# 数据预处理 - 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用PCA
pca = PCA()
# 你可以选择保留的主成分数目，例如保留95%的方差
pca.fit(X_scaled)

# 确定保留多少主成分能解释95%的方差
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
target_variance = 0.95
n_components = np.where(cumulative_variance >= target_variance)[0][0] + 1
print(f"为了保留{target_variance * 100}%的方差，需要{n_components}个主成分。")

# 重新进行PCA，指定保留的主成分数目
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(X_scaled)

# 可视化前两个主成分（如果n_components >= 2）
if n_components >= 2:
    plt.figure()
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Target Class')
    plt.title('PCA of PPPLUS Dataset')
    plt.show()

# # 输出每个性质在第一个主成分上的贡献度（如果需要查看）
# component_importances = np.abs(pca.components_[0])
# for feature, importance in zip(features, component_importances):
#     print(f"Feature: {feature}, Importance: {importance}")

component_importances = np.abs(pca.components_[0])
feature_importances = sorted(zip(features, component_importances), key=lambda x: x[1], reverse=True)

# 打印排序后的特征及其贡献度
for feature, importance in feature_importances:
    print(f"Feature: {feature}, Importance: {importance}")