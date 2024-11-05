# Created by Christian Huyghe on 10/27/2024
# Tests regression strategies with varying parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest

from time import time

X = pd.read_csv('../Data/features.csv')
y = pd.read_csv('../Data/target.csv')

# Initializing Data
kfold = KFold(n_splits=5, random_state=20, shuffle=True)
scaler = StandardScaler()
norm_X = scaler.fit_transform(X)

# Outlier detection and removal
iforest = IsolationForest(contamination=0.001, random_state=20)
combined_data = np.column_stack((norm_X, y - np.mean(y, axis=0) / np.std(y, axis=0)))
iforest.fit(combined_data)
outliers = iforest.predict(combined_data)
cleaned_X = norm_X[outliers == 1]
cleaned_y = y[outliers == 1]


# Calculating PCA of data
pca = decomposition.PCA(n_components=22)
transformed_X = pca.fit_transform(cleaned_X)


# Testing Multiple Linear Regression
regression = LinearRegression()

cross_validation = cross_validate(regression, transformed_X, cleaned_y, cv=kfold, scoring=('neg_root_mean_squared_error', 'r2'))

print("Linear Regression mean R^2 for test data with PCA:", cross_validation["test_r2"].mean())
print("Linear Regression R^2 standard deviation for test data with PCA:", cross_validation["test_r2"].std())
print("Linear Regression mean RMSE for test data with PCA:", cross_validation["test_neg_root_mean_squared_error"].mean() * -1)
print("Linear Regression RMSE standard deviation for test data with PCA:", cross_validation["test_neg_root_mean_squared_error"].std())

cross_validation = cross_validate(regression, cleaned_X, cleaned_y, cv=kfold, scoring=('neg_root_mean_squared_error', 'r2'))

print("\nLinear Regression mean R^2 for test data without PCA:", cross_validation["test_r2"].mean())
print("Linear Regression R^2 standard deviation for test data without PCA:", cross_validation["test_r2"].std())
print("Linear Regression mean RMSE for test data without PCA:", cross_validation["test_neg_root_mean_squared_error"].mean() * -1)
print("Linear Regression RMSE standard deviation for test data without PCA:", cross_validation["test_neg_root_mean_squared_error"].std())

data = []
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Testing Random Forest Regression without PCA
t = time()
for n in [10, 50, 100]:
    for depth in [*range(15, 25), None]:
        rfr = RandomForestRegressor(n_estimators=n, random_state=20, n_jobs=100, max_depth=depth)
        scores = cross_validate(rfr, cleaned_X, cleaned_y.to_numpy().ravel(), cv=kfold, scoring=('neg_root_mean_squared_error', 'r2'))
        r2_mean = scores["test_r2"].mean()
        r2_std = scores["test_r2"].std()
        rmse_mean = scores["test_neg_root_mean_squared_error"].mean() * -1
        rmse_std = scores["test_neg_root_mean_squared_error"].std()
        data.append([n, depth, r2_mean, r2_std, rmse_mean, rmse_std])

print(f"\n{time() - t:2} seconds without PCA")

print("Results (without PCA)")
print(pd.DataFrame(data, columns=["n_estimators", "max_depth", "R^2 Mean", "R^2 STDEV", "RMSE Mean", "RMSE STDEV"]))


# Testing Random Forest Regression with PCA
data = []
t = time()
for n in [10, 50, 100]:
    for depth in [*range(20, 40), None]:
        rfr = RandomForestRegressor(n_estimators=n, random_state=20, n_jobs=100, max_depth=depth)
        scores = cross_validate(rfr, transformed_X, cleaned_y.to_numpy().ravel(), cv=kfold, scoring=('neg_root_mean_squared_error', 'r2'))
        r2_mean = scores["test_r2"].mean()
        r2_std = scores["test_r2"].std()
        rmse_mean = scores["test_neg_root_mean_squared_error"].mean() * -1
        rmse_std = scores["test_neg_root_mean_squared_error"].std()
        data.append([n, depth, r2_mean, r2_std, rmse_mean, rmse_std])

print(f"\n{time() - t:2} seconds with PCA")

print("Results (with PCA)")
print(pd.DataFrame(data, columns=["n_estimators", "max_depth", "R^2 Mean", "R^2 STDEV", "RMSE Mean", "RMSE STDEV"]))


X = pd.read_csv('../Cleaned Data/cleaned_features.csv')
y = pd.read_csv('../Cleaned Data/cleaned_target.csv')

pca = decomposition.PCA(n_components=22)
X = pca.fit_transform(X)

rfr = RandomForestRegressor(n_estimators=100, random_state=20, n_jobs=100, max_depth=35)
rfr.fit(X, y.to_numpy().ravel())
residuals = rfr.predict(X)-y.to_numpy().reshape(1, -1)[0]
plt.scatter(y, residuals)
plt.xlabel("Critical Temperature (Kelvin)")
plt.ylabel("Residual magnitude (Kelvin)")
plt.title("Distribution of residuals")
plt.show()
