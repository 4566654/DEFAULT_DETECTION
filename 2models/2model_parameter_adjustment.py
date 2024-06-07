import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import lightgbm as lgb
import itertools


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
# X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)

# 对X2, Y2进行训练集和测试集划分
X1_train, X1_test, y1_train, y1_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)

positive_ratio = np.sum(y1_train == 1) / np.sum(y1_train == 0)
k = 1 / positive_ratio
# 定义参数网格
params_to_test = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [16, 31, 64, 128],
    'max_depth': [-1, 5, 10, 15],
    'n_estimators': [100, 150, 200, 300],
    'min_child_samples': [10, 20, 30, 50],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0.01, 0.05, 0.1, 0.2],
    'reg_lambda': [0.01, 0.05, 0.1, 0.2]
}
best_f1 = 0
best_params = {}

# 添加class_weight到每次迭代的参数中
for params in itertools.product(*params_to_test.values()):
    params_dict = dict(zip(params_to_test.keys(), params))
    params_dict['class_weight'] = {0: 1, 1: k}  # 添加class_weight
    model = lgb.LGBMClassifier(random_state=42, boosting_type='gbdt', importance_type='gain', **params_dict)

    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=5)
    cv_scores = []

    for train_idx, val_idx in skf.split(X1_train, y1_train):
        X_train_fold, X_val_fold = X1_train[train_idx], X1_train[val_idx]
        y_train_fold, y_val_fold = y1_train[train_idx], y1_train[val_idx]

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        cv_scores.append(f1_score(y_val_fold, preds, average='weighted'))

    mean_f1 = np.mean(cv_scores)
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_params = params_dict.copy()

print("Best parameters found manually: ", best_params)
print("Best mean F1 score: ", best_f1)