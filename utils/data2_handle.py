import pandas as pd
import numpy as np

data = pd.read_excel('../data/data2.xlsx')
X = data.iloc[:, :].values
features = data.columns[:].values

# 打印原表头中的特定位置以验证
print(features[5])
print(features[7])

# 定义要移除的列索引
columns_to_remove = [5, 7]

# 移除指定列索引对应的表头名称
# features = np.delete(features, columns_to_remove)

# 初始化四个列表用于存储数据（这部分逻辑不变）
X1, X2 = [], []

for i, x in enumerate(X):
    # temp_x = np.delete(x, columns_to_remove)
    # if (x[7] == 5 or x[7] == 6) and x[5] == 18:
    #     X1.append(temp_x)
    # else:
    #     X2.append(temp_x)
    if (x[7] == 5 or x[7] == 6) and x[5] == 18:
        X1.append(x)
    else:
        X2.append(x)

X1 = np.array(X1)
X2 = np.array(X2)

# 将NumPy数组转换为DataFrame，并设置正确的列名
df_X1 = pd.DataFrame(X1, columns=features)
df_X2 = pd.DataFrame(X2, columns=features)

excel_file = '../data/data2_selected.xlsx'

# 将两个DataFrame保存到同一个Excel文件的不同工作表，并保留正确的表头
with pd.ExcelWriter(excel_file) as writer:
    df_X1.to_excel(writer, sheet_name='Sheet1', index=False)
    df_X2.to_excel(writer, sheet_name='Sheet2', index=False)

print(f"Data has been successfully saved to {excel_file} with headers.")