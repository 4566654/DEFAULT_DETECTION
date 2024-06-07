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
features = data1.columns[:].values

# 定义要移除的列索引
columns_to_remove = [0, 1, 2, 3, 4, 5, 7, 5, 3, 54, 55, 56, 57, 58, 59]

X1 = np.delete(X1, columns_to_remove, axis=1)
X2 = np.delete(X2, columns_to_remove, axis=1)
features = np.delete(features, columns_to_remove)

# 对X1, Y1进行训练集和测试集划分
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)

# 对X2, Y2进行训练集和测试集划分
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)

# 分别训练两个LGBM模型
model1 = LGBMClassifier(random_state=42, learning_rate=0.01, num_leaves=30, max_depth=6, n_estimators=100,
                        min_child_samples=16, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                        min_split_gain=0.01, min_child_weight=0.001, importance_type='gain', boosting_type='gbdt', class_weight={0:1, 1:50})
model1.fit(X1_train, y1_train)

model2 = LGBMClassifier(random_state=42, learning_rate=0.01, num_leaves=30, max_depth=6, n_estimators=100,
                        min_child_samples=16, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                        min_split_gain=0.01, min_child_weight=0.001, importance_type='gain', boosting_type='gbdt', class_weight={0:1, 1:50})
model2.fit(X2_train, y2_train)

# 分别进行预测
y1_pred = model1.predict(X1_test)
y2_pred = model2.predict(X2_test)

y_pred = np.concatenate((y1_pred, y2_pred))
y_test = np.concatenate((y1_test, y2_test))

print("------------------------模型1----------------------------")
matrix1 = confusion_matrix(y1_test, y1_pred)
print("Model 1's Confusion Matrix:\n", matrix1)
print("违约类的recall for Model 1:", matrix1[1, 1] / (matrix1[1, 1] + matrix1[1, 0]))
print("违约类的precision for Model 1:", matrix1[1, 1] / (matrix1[1, 1] + matrix1[0, 1]))

f1 = f1_score(y1_test, y1_pred)

print(f"F1 Score: {f1}")

print("------------------------模型2----------------------------")
matrix2 = confusion_matrix(y2_test, y2_pred)
print("Model 2's Confusion Matrix:\n", matrix2)
print("违约类的recall for Model 2:", matrix2[1, 1] / (matrix2[1, 1] + matrix2[1, 0]))
print("违约类的precision for Model 2:", matrix2[1, 1] / (matrix2[1, 1] + matrix2[0, 1]))

f1 = f1_score(y2_test, y2_pred)

print(f"F1 Score: {f1}")


print("------------------------总模型----------------------------")
matrix = confusion_matrix(y_test, y_pred)
print("Model's Confusion Matrix:\n", matrix)
print("违约类的recall for Model:", matrix[1, 1] / (matrix[1, 1] + matrix[1, 0]))
print("违约类的precision for Model:", matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]))

f1 = f1_score(y_test, y_pred)

print(f"F1 Score: {f1}")

# 定义评估指标和交叉验证折叠数
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_folds = 5

# 模型定义保持不变，但我们将直接在交叉验证中使用它们

# 对模型1进行交叉验证
results_model1 = cross_validate(model1, X1, Y1, cv=cv_folds, scoring=scoring)
print("\n------------------------模型1交叉验证结果----------------------------")
print("Accuracy: ", results_model1['test_accuracy'].mean())
print("Precision: ", results_model1['test_precision'].mean())
print("Recall: ", results_model1['test_recall'].mean())
print("F1 Score: ", results_model1['test_f1'].mean())

# 对模型2进行交叉验证
results_model2 = cross_validate(model2, X2, Y2, cv=cv_folds, scoring=scoring)
print("\n------------------------模型2交叉验证结果----------------------------")
print("Accuracy: ", results_model2['test_accuracy'].mean())
print("Precision: ", results_model2['test_precision'].mean())
print("Recall: ", results_model2['test_recall'].mean())
print("F1 Score: ", results_model2['test_f1'].mean())