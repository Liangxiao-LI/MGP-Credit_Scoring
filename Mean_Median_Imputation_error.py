#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
#%%
inputfile = 'Final_Filtered_DataV2.xlsx'
df = pd.read_excel(inputfile, engine='openpyxl')

# %%
# Filter the DataFrame to include only rows where 'NumberOfDependents' is not NA
df_filtered = df[df['NumberOfDependents'].notna()]
NOD = df_filtered['NumberOfDependents']

df_filtered = df[df['MonthlyIncome'].notna()]
MI = df_filtered['MonthlyIncome']


# Using .values to convert the series to a NumPy array
NOD_vector = NOD.values
MI_vector = MI.values

for i in range(len(MI_vector)):
    if MI_vector[i] != 0:
        MI_vector[i] = np.log(MI_vector[i])

# %%

# Calculate the mean and median for NOD_vector and MI_vector
NOD_mean = np.mean(NOD_vector)
NOD_median = np.median(NOD_vector)

MI_mean = np.mean(MI_vector)
MI_median = np.median(MI_vector)

NOD_mean_vector = np.full(len(NOD_vector), NOD_mean)
NOD_median_vector = np.full(len(NOD_vector), NOD_median)
MI_mean_vector = np.full(len(MI_vector), MI_mean)
MI_median_vector = np.full(len(MI_vector), MI_median)


# %% NOD

# Calculate Mean Square Error (MSE)
mse_mean,mse_median = mean_squared_error(NOD, NOD_mean_vector),mean_squared_error(NOD, NOD_median_vector)
# Calculate R-squared
r_mean,r_median = r2_score(NOD, NOD_mean_vector),r2_score(NOD, NOD_median_vector)
# Calculate Mean Absolute Error (MAE)
mae_mean,mae_median = mean_absolute_error(NOD, NOD_mean_vector),mean_absolute_error(NOD, NOD_median_vector)

print(mse_mean,mse_median)
print(mae_mean,mae_median)
print(r_mean,r_median)
# %% MI

# Calculate Mean Square Error (MSE)
mse_mean,mse_median = mean_squared_error(MI, MI_mean_vector),mean_squared_error(MI, MI_median_vector)
# Calculate R-squared
r_mean,r_median = r2_score(MI, MI_mean_vector),r2_score(MI, MI_median_vector)
# Calculate Mean Absolute Error (MAE)
mae_mean,mae_median = mean_absolute_error(MI, MI_mean_vector),mean_absolute_error(MI, MI_median_vector)

print(mse_mean,mse_median)
print(mae_mean,mae_median)
print(r_mean,r_median)


# %%
