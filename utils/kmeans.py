import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def kmeans_xlsx(data_path, sheet_name, columns):
    # 读取Excel文件和数据预处理部分保持不变
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    columns_to_cluster = df.iloc[:, columns]
    features = df.columns[columns].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(columns_to_cluster)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(scaled_data)
    df['Cluster'] = kmeans.labels_

    # 绘制三维聚类图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']

    for i in range(kmeans.n_clusters):  # 修改循环以适应所有聚类
        # 使用列名提取属于该类别的数据点
        class_members = df[df['Cluster'] == i]
        ax.scatter(class_members[features[0]], class_members[features[1]],
                   class_members[features[2]],  # 确保使用列名
                   c=colors[i], label=f'Cluster {i}')

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.legend()
    plt.title('3D Visualization of KMeans Clustering')
    plt.show()

    # 将两个DataFrame保存到同一个Excel文件的不同工作表，并保留正确的表头

    return df

if __name__ == '__main__':
    data_path = '../data/data2_selected.xlsx'
    sheet_name = 'Sheet1'
    output_path = '../data/data2_selected_kmeans.xlsx'
    columns = [55, 56, 57]
    df1 = kmeans_xlsx(data_path, sheet_name, columns)

    sheet_name = 'Sheet2'
    df2 = kmeans_xlsx(data_path, sheet_name, columns)

    with pd.ExcelWriter(output_path) as writer:
        df1.to_excel(writer, sheet_name='sheet1', index=False)
        df2.to_excel(writer, sheet_name='sheet2', index=False)
