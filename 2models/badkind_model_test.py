import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate


data2 = pd.read_excel('../data/data2_final.xlsx', sheet_name='sheet2')
X2 = data2.iloc[:, :].values
Y2 = data2.iloc[:, 58].values

# 定义要移除的列索引
columns_to_remove = [0, 1, 2, 3, 4, 5, 7, 5, 3, 54, 55, 56, 57, 58, 59]

X2 = np.delete(X2, columns_to_remove, axis=1)

# 对X2, Y2进行训练集和测试集划分
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)


positive_ratio = np.sum(y2_train == 1) / np.sum(y2_train == 0)
k = 1 / positive_ratio
model2 = LGBMClassifier(
        random_state=42,  # 保持随机种子以确保结果可复现
        learning_rate=0.05,  # 适中的学习率，适用于多数情况
        num_leaves=100,  # 增加叶子节点数以提高模型容量，31是一个常见的默认选择
        max_depth=-1,  # 不限制树的最大深度，让算法自动决定，或根据数据特性设定
        n_estimators=500,  # 提升树的数量，100是一个常用的起始点，可根据需要增加
        min_child_samples=20,  # 每个叶子节点的最小样本数，有助于防止过拟合
        subsample=0.8,  # 子样本比例，保持默认的较高效能和防止过拟合之间的平衡
        colsample_bytree=0.9,  # 列子采样比例，同样为了平衡性能与泛化能力
        reg_alpha=10,  # L1正则化项，轻度正则化以避免过拟合
        reg_lambda=10,  # L2正则化项，与reg_alpha相似，可根据实际情况调整
        min_split_gain=0,  # 允许分裂只要有增益，保持灵活性
        min_child_weight=0.001,  # 较小的值允许创建较轻的叶子节点
        importance_type='gain',  # 使用增益作为特征重要性度量
        boosting_type='gbdt',  # 使用梯度提升决策树作为默认的提升类型
        class_weight={0: 1, 1: k},  # 如果数据均衡，可以不设置class_weight，或根据实际情况调整
    )
model2.fit(X2_train, y2_train)

y2_pred = model2.predict(X2_test)

# 定义评估指标和交叉验证折叠数
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_folds = 5

# 模型定义保持不变，但我们将直接在交叉验证中使用它们

# 对模型2进行交叉验证
results_model2 = cross_validate(model2, X2, Y2, cv=cv_folds, scoring=scoring)
f12 = results_model2['test_f1'].mean()
print("\n------------------------模型2交叉验证结果----------------------------")
print("Accuracy: ", results_model2['test_accuracy'].mean())
print("Precision: ", results_model2['test_precision'].mean())
print("Recall: ", results_model2['test_recall'].mean())
print("F1 Score: ", results_model2['test_f1'].mean())


