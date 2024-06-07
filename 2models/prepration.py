import pandas as pd
import numpy as np
from scipy.spatial import distance

# 加载数据
data1 = pd.read_excel('../data/data2_selected_kmeans.xlsx', sheet_name='sheet1')
data2 = pd.read_excel('../data/data2_selected_kmeans.xlsx', sheet_name='sheet2')

# 从data1中筛选'cluster'列值为1的行
data1_cluster_1 = data1[data1['Cluster'] == 1]

# 转换为NumPy数组
data1_cluster_1_np = data1_cluster_1.values

# 计算指定列的最小值，得到三维向量
min_vector = np.min(data1_cluster_1_np[:, [55, 56, 57]], axis=0)

# 计算所有行的距离
distances = np.array([distance.euclidean(row[55:58], min_vector) for row in data1_cluster_1_np])

# 根据距离排序索引
sorted_indices = distances.argsort()

# 分割数据：前40%和后60%
split_index = int(len(sorted_indices) * 0.4)
top_40_indices = sorted_indices[:split_index]
buttom_60_indices = sorted_indices[split_index:]

# 根据索引选择数据
top_40_data_np = data1_cluster_1_np[top_40_indices]
buttom_60_data_np = data1_cluster_1_np[buttom_60_indices]

# 将NumPy数组转换回DataFrame
top_40_data_df = pd.DataFrame(top_40_data_np, columns=data1_cluster_1.columns)
buttom_60_data_df = pd.DataFrame(buttom_60_data_np, columns=data1_cluster_1.columns)

# 合并数据到data2和更新data1
data2 = pd.concat([data2, top_40_data_df], ignore_index=True)
data1 = data1[data1['Cluster'] != 1]
data1 = pd.concat([data1, buttom_60_data_df], ignore_index=True)

excel_file = '../data/data2_final.xlsx'

# 保存到Excel
with pd.ExcelWriter(excel_file) as writer:
    data1.to_excel(writer, sheet_name='sheet1', index=False)
    data2.to_excel(writer, sheet_name='sheet2', index=False)