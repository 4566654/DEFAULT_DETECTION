import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate


data1 = pd.read_excel('../data/data2_final.xlsx', sheet_name='sheet1')
data2 = pd.read_excel('../data/data2_final.xlsx', sheet_name='sheet2')
X1 = data1.iloc[:, :].values
X2 = data2.iloc[:, :].values
Y1 = data1.iloc[:, 58].values
Y2 = data2.iloc[:, 58].values

# 定义要移除的列索引
columns_to_remove = [0, 1, 2, 3, 4, 5, 7, 5, 3, 54, 55, 56, 57, 58, 59]

X1 = np.delete(X1, columns_to_remove, axis=1)
X2 = np.delete(X2, columns_to_remove, axis=1)

# 对X1, Y1进行训练集和测试集划分
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)

# 对X2, Y2进行训练集和测试集划分
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)

f11_tmp=0
f12_tmp=0
for k in range(1, 100):
    # 分别训练两个LGBM模型
    model1 = LGBMClassifier(
        random_state=42,  # 保持随机种子以确保结果可复现
        learning_rate=0.1,  # 适中的学习率，适用于多数情况
        num_leaves=31,  # 增加叶子节点数以提高模型容量，31是一个常见的默认选择
        max_depth=-1,  # 不限制树的最大深度，让算法自动决定，或根据数据特性设定
        n_estimators=150,  # 提升树的数量，100是一个常用的起始点，可根据需要增加
        min_child_samples=20,  # 每个叶子节点的最小样本数，有助于防止过拟合
        subsample=0.8,  # 子样本比例，保持默认的较高效能和防止过拟合之间的平衡
        colsample_bytree=0.8,  # 列子采样比例，同样为了平衡性能与泛化能力
        reg_alpha=0.1,  # L1正则化项，轻度正则化以避免过拟合
        reg_lambda=0.1,  # L2正则化项，与reg_alpha相似，可根据实际情况调整
        min_split_gain=0,  # 允许分裂只要有增益，保持灵活性
        min_child_weight=0.001,  # 较小的值允许创建较轻的叶子节点
        importance_type='gain',  # 使用增益作为特征重要性度量
        boosting_type='gbdt',  # 使用梯度提升决策树作为默认的提升类型
        class_weight={0: 1, 1: k},  # 如果数据均衡，可以不设置class_weight，或根据实际情况调整
    )
    model1.fit(X1_train, y1_train)

    model2 = LGBMClassifier(
        random_state=42,  # 保持随机种子以确保结果可复现
        learning_rate=0.1,  # 适中的学习率，适用于多数情况
        num_leaves=31,  # 增加叶子节点数以提高模型容量，31是一个常见的默认选择
        max_depth=-1,  # 不限制树的最大深度，让算法自动决定，或根据数据特性设定
        n_estimators=150,  # 提升树的数量，100是一个常用的起始点，可根据需要增加
        min_child_samples=20,  # 每个叶子节点的最小样本数，有助于防止过拟合
        subsample=0.8,  # 子样本比例，保持默认的较高效能和防止过拟合之间的平衡
        colsample_bytree=0.8,  # 列子采样比例，同样为了平衡性能与泛化能力
        reg_alpha=0.1,  # L1正则化项，轻度正则化以避免过拟合
        reg_lambda=0.1,  # L2正则化项，与reg_alpha相似，可根据实际情况调整
        min_split_gain=0,  # 允许分裂只要有增益，保持灵活性
        min_child_weight=0.001,  # 较小的值允许创建较轻的叶子节点
        importance_type='gain',  # 使用增益作为特征重要性度量
        boosting_type='gbdt',  # 使用梯度提升决策树作为默认的提升类型
        class_weight={0: 1, 1: k},  # 如果数据均衡，可以不设置class_weight，或根据实际情况调整
    )
    model2.fit(X2_train, y2_train)

    # 分别进行预测
    y1_pred = model1.predict(X1_test)
    y2_pred = model2.predict(X2_test)

    y_pred = np.concatenate((y1_pred, y2_pred))
    y_test = np.concatenate((y1_test, y2_test))

    # 定义评估指标和交叉验证折叠数
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_folds = 5

    # 模型定义保持不变，但我们将直接在交叉验证中使用它们

    # 对模型1进行交叉验证
    results_model1 = cross_validate(model1, X1, Y1, cv=cv_folds, scoring=scoring)
    f11 = results_model1['test_f1'].mean()
    if f11 > f11_tmp:
        f11_tmp = f11
        k1_final = k
    print("\n------------------------模型1交叉验证结果----------------------------")
    print("Accuracy: ", results_model1['test_accuracy'].mean())
    print("Precision: ", results_model1['test_precision'].mean())
    print("Recall: ", results_model1['test_recall'].mean())
    print("F1 Score: ", results_model1['test_f1'].mean())

    # 对模型2进行交叉验证
    results_model2 = cross_validate(model2, X2, Y2, cv=cv_folds, scoring=scoring)
    f12 = results_model2['test_f1'].mean()
    if f12 > f12_tmp:
        f12_tmp = f12
        k2_final = k
    print("\n------------------------模型2交叉验证结果----------------------------")
    print("Accuracy: ", results_model2['test_accuracy'].mean())
    print("Precision: ", results_model2['test_precision'].mean())
    print("Recall: ", results_model2['test_recall'].mean())
    print("F1 Score: ", results_model2['test_f1'].mean())

print(k1_final)
print(f11_tmp)
print(k2_final)
print(f12_tmp)
