import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
import shap
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
data = pd.read_excel('data/ppplus(1).xlsx')
X = data.iloc[:, 2:29].values
y = data.iloc[:, 29].values
features = data.columns[2:29].values

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. smote 处理
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(pd.Series(y_train).value_counts())

# 4. 模型训练 使用LBGM模型
# 调节学习率
model = LGBMClassifier(random_state=42, learning_rate=0.01, num_leaves=30, max_depth=6, n_estimators=100,
                       min_child_samples=16, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                       min_split_gain=0.01, min_child_weight=0.001, importance_type='gain', boosting_type='gbdt')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# 使用seaborn绘制热图表示混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Predicted 0', 'Predicted 1'],
#             yticklabels=['Actual 0', 'Actual 1'])
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix')
# plt.show()


print("违约类的recall:", matrix[1, 1] / (matrix[1, 1] + matrix[1, 0]))
print("违约类的precision:", matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]))

# # 6. shap 分析
# # 确保htmls文件夹存在
# if not os.path.exists('shap'):
#     os.makedirs('shap')
# if not os.path.exists('shap/all_htmls'):
#     os.makedirs('shap/all_htmls')
# if not os.path.exists('shap/default_htmls'):
#     os.makedirs('shap/default_htmls')
# if not os.path.exists('shap/nodefault_htmls'):
#     os.makedirs('shap/nodefault_htmls')
# # 创建SHAP解释器
# explainer = shap.TreeExplainer(model)
#
# # 计算SHAP值
# shap_values = explainer.shap_values(X_train)
#
# # 可视化SHAP值
# shap.summary_plot(shap_values, X_train, feature_names=features)
#
# plt.savefig('shap/shap_summary_plot.png')
#
# # 确保matplotlib关闭图像显示窗口，如果你在脚本中运行这行可能需要
# plt.close()
# shap.initjs()
#
# # 保存为HTML文件
# # shap.save_html("force_plot.html", shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:], feature_names=features))
#
# # 循环遍历测试集，为每个样本生成并保存force_plot
# for i in range(len(y_pred)):
#     if (y_pred[i] == 1):
#         # 为当前样本生成force_plot并保存到'shap/default_htmls'文件夹中
#         plot_filename = os.path.join('shap/default_htmls', f"force_plot_default_{i}.html")
#         shap.save_html(plot_filename, shap.force_plot(explainer.expected_value, shap_values[i, :], X_test[i, :], feature_names=features))
#         print(f"Force plot for sample {i} saved as {plot_filename}")
#     else:
#         # 为当前样本生成force_plot并保存到'shap/nodefault_htmls'文件夹中
#         plot_filename = os.path.join('shap/nodefault_htmls', f"force_plot_nodefault_{i}.html")
#         shap.save_html(plot_filename, shap.force_plot(explainer.expected_value, shap_values[i, :], X_test[i, :], feature_names=features))
#         print(f"Force plot for sample {i} saved as {plot_filename}")
#     # 为当前样本生成force_plot并保存到'shap/all_htmls'文件夹中
#     plot_filename = os.path.join('shap/all_htmls', f"force_plot_{i}.html")
#     shap.save_html(plot_filename, shap.force_plot(explainer.expected_value, shap_values[i, :], X_test[i, :], feature_names=features))
#     print(f"Force plot for sample {i} saved as {plot_filename}")
