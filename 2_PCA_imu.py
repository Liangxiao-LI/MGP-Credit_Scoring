# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 01:04:02 2024

@author: Evelyn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载数据集
data = pd.read_excel('Imputed_data.xlsx')

# 删除缺失值所在的行
data = data.dropna()

# 提取特征列
X = data.drop(columns=['Index'])

# 标准化数据
X_centered = (X - X.mean()) / X.std()

# 执行PCA
pca = PCA()
X_pca = pca.fit_transform(X_centered)

# 计算特征值
eigenvalues = pca.explained_variance_

# 绘制Scree图
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', color='skyblue')
plt.axhline(y=1, linestyle='--', color='red', label='Eigenvalue = 1')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.title('Scree Plot')
plt.legend()
plt.grid(True)
plt.show()
#%%
# 应用Kaiser准则选择主成分数量
num_components_to_keep = np.sum(eigenvalues > 1)

print("Number of principal components to retain for PCA based on Kaiser criterion:", num_components_to_keep)
