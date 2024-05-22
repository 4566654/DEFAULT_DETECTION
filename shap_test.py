import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import shap

import matplotlib.pyplot as plt

# 指定字体名称和路径
font_name = 'SimHei'  # 例如，使用黑体
font_path = 'C:/Windows/Fonts/SimHei.ttf'  # 字体文件的路径

# 注册字体
plt.rcParams['font.family'] = font_name
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

# 如果需要，可以将字体文件添加到matplotlib的字体缓存中
import matplotlib.font_manager as font_manager
font_manager.fontManager.addfont(font_path)

# 1. 读取数据
data = pd.read_excel('plus.xlsx')
X = data.iloc[:, 2:43].values
y = data.iloc[:, 43].values
features = data.columns[2:43].values

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. smote 处理
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(pd.Series(y_train).value_counts())

# 4. 模型训练 使用逻辑回归模型
# 调节学习率
model = LogisticRegression(max_iter=10000, C=1.0, random_state=42)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# # 5. 输出模型评分
# features_importance = model.feature_importances_
# features_importance_rank = []
# for i in range(len(features)):
#     features_importance_rank.append((features[i], features_importance[i]))
# features_importance_rank_sorted = sorted(features_importance_rank, key=lambda x: x[1], reverse=True)
# for i in range(len(features)):
#     print(features_importance_rank_sorted[i])
# matrix = confusion_matrix(y_test, y_pred)
# print(matrix)
# # a = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
# # b = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
# # print("UAR:", (a + b) / 2)
# print("违约类的recall:", matrix[1, 1] / (matrix[1, 1] + matrix[1, 0]))
# print("违约类的precision:", matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]))

# # 6. shap 分析
# # 创建一个树形模型解释器
# explainer = shap.TreeExplainer(model)
#
# # 获取 SHAP 值
# shap_values = explainer.shap_values(X_train)
#
# # 绘制 summary plot
# shap.summary_plot(shap_values, X_train, feature_names=features)

# # 6. shap 分析
# # 创建一个SHAP解释器
# explainer = shap.TreeExplainer(model)
#
# # 计算SHAP值
# shap_values = explainer.shap_values(X_train)
#
# # 可视化SHAP值
# shap.summary_plot(shap_values, X_train, feature_names=features)


# 6. shap 分析
# 创建一个SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算SHAP值
# 对于二分类任务，shap_values将返回一个列表，其中包含每个类的SHAP值
shap_values = explainer.shap_values(X_train)

# 选择正类的SHAP值
shap_values_pos = shap_values[1]
print(type(shap_values_pos))

# 可视化SHAP值
shap.summary_plot(shap_values, X_train, feature_names=features)

#查看单个样本的特征贡献的第一种方法
shap.initjs()# colab需要在每个cell上运行这个命令，如果你是jupyter notebook或jupyter lab可以把这行注释掉
shap.plots.force(explainer.expected_value[1], shap_values[1][0], X_train)