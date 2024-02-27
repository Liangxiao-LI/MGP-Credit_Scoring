# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:37:51 2024

@author: Evelyn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 加载数据集
data = pd.read_excel('Final_Filtered_Data_V2_XGB.xlsx')

# 删除'Index'列
data = data.drop(columns=['Index'])

# 提取自变量
X = data.drop(columns=['SeriousDlqin2yrs'])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
percentile=95
# 并行分析函数
def parallel_analysis(X, iterations=1000, percentile=95):
    n, p = X.shape
    eigenvalues_actual, _ = np.linalg.eig(np.corrcoef(X.T))
    eigenvalues_random = np.zeros((iterations, p))
    
    for i in range(iterations):
        # 生成随机数据矩阵
        random_data = np.random.normal(size=(n, p))
        # 计算相关系数矩阵
        random_corr_matrix = np.corrcoef(random_data.T)
        # 计算特征值
        eigenvalues, _ = np.linalg.eig(random_corr_matrix)
        eigenvalues_random[i] = eigenvalues
    
    # 计算随机特征值的百分位数
    percentile_value = np.percentile(eigenvalues_random, percentile, axis=0)
    
    # 统计实际特征值大于随机特征值的组件数
    num_components = np.sum(eigenvalues_actual > percentile_value)
    
    return num_components, eigenvalues_actual, percentile_value

# 执行并行分析
num_components, eigenvalues_actual, percentile_value = parallel_analysis(X_scaled)

print("根据并行分析应保留的成分数:", num_components)

# 计算PCA
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_scaled)

# 计算累积方差解释比率
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 打印每个成分的累积方差解释比率
print("每个成分的累积方差解释比率:")
for i, var in enumerate(cumulative_variance, start=1):
    print(f"成分 {i}: {var:.4f}")

# 绘制特征值大于随机特征值的比较图和累积方差解释比率
plt.figure(figsize=(12, 6))

# 绘制特征值比较图
plt.subplot(1, 2, 1)
plt.plot(range(1, len(eigenvalues_actual) + 1), eigenvalues_actual, label='实际特征值')
plt.plot(range(1, len(percentile_value) + 1), percentile_value, label=f'{percentile}th 百分位数的随机特征值')
plt.title('实际特征值与随机特征值的比较')
plt.xlabel('组件数')
plt.ylabel('特征值')
plt.legend()
plt.grid(True)

# 绘制累积方差解释比率图
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.title('累积方差解释比率')
plt.xlabel('成分数')
plt.ylabel('累积方差解释比率')
plt.grid(True)

plt.tight_layout()
plt.show()
