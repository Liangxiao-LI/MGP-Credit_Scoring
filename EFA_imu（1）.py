# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 08:55:41 2024

@author: Evelyn
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 导入数据
data = pd.read_excel('Imputed_data.xlsx')

# 删除无用列
df = data.drop(columns=['Index'])

# 标准化数据
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
#%%
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# 计算 KMO 值
kmo_all, kmo_model = calculate_kmo(df)


# 打印结果
print("KMO metric:", kmo_model)


#%%
# 因子分析
factor_analysis = FactorAnalysis(n_components=df.shape[1], random_state=42)
factor_analysis.fit(df_scaled)

# 计算因子载荷矩阵
loadings = factor_analysis.components_

# 计算每个因子解释的方差
eigenvalues = np.diag(np.dot(loadings, loadings.T))

# 绘制Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1')
plt.title('Scree Plot')
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue')
plt.legend()
plt.grid(True)
plt.show()

# 根据Scree Plot确定因子数量
num_factors = np.sum(eigenvalues > 1)
print("根据Scree Plot确定的因子数量:", num_factors)

# 提取PCA的结果
loadings_correlation = pd.DataFrame(factor_analysis.components_.T, columns=['Factor{}'.format(i+1) for i in range(factor_analysis.n_components)], index=df.columns)
explained_variance_ratio_correlation = np.var(df_scaled, axis=0) / np.sum(np.var(df_scaled, axis=0))
cumulative_variance_ratio_correlation = explained_variance_ratio_correlation.cumsum()

# 计算h2、u2和com
h2 = np.square(loadings_correlation).sum(axis=1)
u2 = 1 - h2
com = h2 / u2

# 输出加载项、方差解释比例和h2 u2 com
print("Factor loadings:")
print(loadings_correlation)
print("\nProportion of Variance:")
print(explained_variance_ratio_correlation)
print("\nCumulative Proportion of Variance:")
print(cumulative_variance_ratio_correlation)
print("\nh2:")
print(h2)
print("\nu2:")
print(u2)
print("\ncom:")
print(com)
