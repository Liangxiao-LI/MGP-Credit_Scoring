# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 23:49:14 2024

@author: Evelyn
"""

#% Importing Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
data = pd.read_excel('Imputed_data.xlsx')
# Drop the 'Unnamed: 0' column
#df = data.drop(columns=['Index: 0'])
# 删除含有缺失值的行
#data = df.dropna()
print(data.info())
#%%
df = data.drop(columns=['Index'])
print(df.info())

#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 删除含有缺失值的行
#X = df.dropna()

# 1. 数据标准化
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(df)

# 2. 计算相关系数矩阵
correlation_matrix = pd.DataFrame(df).corr()


# 3. 创建并拟合基于相关系数矩阵的PCA模型
pca_correlation = PCA(n_components=9)  # 设置为原始特征的数量
pca_correlation.fit(correlation_matrix)

# 提取PCA的结果
loadings_correlation = pd.DataFrame(pca_correlation.components_.T, columns=['PC{}'.format(i+1) for i in range(9)], index=df.columns)
explained_variance_ratio_correlation = pca_correlation.explained_variance_ratio_
cumulative_variance_ratio_correlation = explained_variance_ratio_correlation.cumsum()

# 计算h2、u2和com
h2 = np.square(loadings_correlation).sum(axis=1)
u2 = 1 - h2
com = h2 / u2

# 输出加载项、方差解释比例和h2 u2 com
print("Standardized loadings (pattern matrix) based upon correlation matrix")
print(loadings_correlation)
print("\nProportion Var")
print(explained_variance_ratio_correlation)
print("\nCumulative Var")
print(cumulative_variance_ratio_correlation)
print("\nh2")
print(np.array([h2]))
print("\nu2")
print(np.array([u2]))
print("\ncom")
print(np.array([com]))

#%%
# 提取com数组的值
com_values = com.values

# 计算平均项复杂度
mean_item_complexity = np.mean(com_values)

# 打印平均项复杂度
print("Mean Item Complexity:", mean_item_complexity)

#%%
# 使用Kaiser准则确定主成分数量
sufficient_components_kaiser = np.sum(explained_variance_ratio_correlation > 1)
print("Sufficient Components (Kaiser Criterion):", sufficient_components_kaiser)
#%%

# 绘制Scree Plot
plt.plot(range(1, len(explained_variance_ratio_correlation) + 1), explained_variance_ratio_correlation, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()
#%%
print("Explained Variance Ratio (Scree Plot):")
for i, explained_variance in enumerate(explained_variance_ratio_correlation):
    print("PC{}: {:.4f}".format(i + 1, explained_variance))

#%%
# 计算解释方差比例的变化率
variance_change = np.diff(explained_variance_ratio_correlation)

# 找到拐点
elbow_index = np.argmax(variance_change) + 1

print("拐点在PC{}.".format(elbow_index))
