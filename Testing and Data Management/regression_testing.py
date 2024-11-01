# Created by Christian Huyghe on 10/27/2024
# Tests regression strategies with varying parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest

from time import time

X = pd.read_csv('../Data/features.csv')
y = pd.read_csv('../Data/target.csv')

# Initializing Data
train_X, test_X,  train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=20)
scaler = StandardScaler()
norm_train_X = scaler.fit_transform(train_X)
norm_test_X = scaler.transform(test_X)

# Outlier detection and removal
iforest = IsolationForest(contamination=0.001, random_state=20)
combined_data = np.column_stack((norm_train_X, train_y - np.mean(train_y, axis=0) / np.std(train_y, axis=0)))
iforest.fit(combined_data)
outliers = iforest.predict(combined_data)
cleaned_norm_train_X = norm_train_X[outliers == 1]
cleaned_train_y = train_y[outliers == 1]


# Calculating PCA of data
pca = decomposition.PCA(n_components=22)
transformed_train_X = pca.fit_transform(cleaned_norm_train_X)
transformed_test_X = pca.transform(norm_test_X)


# Testing Multiple Linear Regression
regression = LinearRegression()
regression.fit(transformed_train_X, cleaned_train_y)

print("Linear Regression R^2 for training data with PCA:", regression.score(transformed_train_X, cleaned_train_y))
print("Linear Regression RMSE for training data with PCA:", mean_squared_error(regression.predict(transformed_train_X), cleaned_train_y)**.5)

print("Linear Regression R^2 for test data with PCA:", regression.score(transformed_test_X, test_y))
print("Linear Regression RMSE for test data with PCA:", mean_squared_error(regression.predict(transformed_test_X), test_y)**.5)

regression = LinearRegression()
regression.fit(cleaned_norm_train_X, cleaned_train_y)

print("\nLinear Regression R^2 for training data:", regression.score(cleaned_norm_train_X, cleaned_train_y))
print("Linear Regression RMSE for training data:", mean_squared_error(regression.predict(cleaned_norm_train_X), cleaned_train_y)**.5)

print("Linear Regression R^2 for test data:", regression.score(norm_test_X, test_y))
print("Linear Regression RMSE for test data:", mean_squared_error(regression.predict(norm_test_X), test_y)**.5)

data = []
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Testing Random Forest Regression without PCA
t = time()
for n in [10, 50, 100]:
    for depth in [*range(15, 25), None]:
        rfr = RandomForestRegressor(n_estimators=n, random_state=20, n_jobs=100, max_depth=depth)
        rfr.fit(cleaned_norm_train_X, cleaned_train_y.to_numpy().ravel())
        train_R2 = rfr.score(cleaned_norm_train_X, cleaned_train_y)
        train_rmse = mean_squared_error(rfr.predict(cleaned_norm_train_X), cleaned_train_y)**.5
        test_R2 = rfr.score(norm_test_X, test_y)
        test_rmse = mean_squared_error(rfr.predict(norm_test_X), test_y)**.5
        data.append([n, depth, train_R2, train_rmse, test_R2, test_rmse])

print(f"\n{time() - t:2} seconds without PCA")

print("Results (without PCA)")
print(pd.DataFrame(data, columns=["n_estimators", "max_depth", "Training R^2", "Training RMSE", "Test R^2", "Test RMSE"]))


# Testing Random Forest Regression with PCA
data = []
t = time()
for n in [10, 50, 100]:
    for depth in [*range(15, 25), None]:
        rfr = RandomForestRegressor(n_estimators=n, random_state=20, n_jobs=100, max_depth=depth)
        rfr.fit(transformed_train_X, cleaned_train_y.to_numpy().ravel())
        train_R2 = rfr.score(transformed_train_X, cleaned_train_y)
        train_rmse = mean_squared_error(rfr.predict(transformed_train_X), cleaned_train_y)**.5
        test_R2 = rfr.score(transformed_test_X, test_y)
        test_rmse = mean_squared_error(rfr.predict(transformed_test_X), test_y)**.5
        data.append([n, depth, train_R2, train_rmse, test_R2, test_rmse])

print(f"\n{time() - t:2} seconds with PCA")

print("Results (with PCA)")
print(pd.DataFrame(data, columns=["n_estimators", "max_depth", "Training R^2", "Training RMSE", "Test R^2", "Test RMSE"]))

rfr = RandomForestRegressor(n_estimators=50, random_state=20, n_jobs=100, max_depth=23)
rfr.fit(transformed_train_X, cleaned_train_y.to_numpy().ravel())

residuals = rfr.predict(pca.transform(scaler.transform(X)))-y.to_numpy().reshape(1, -1)[0]
plt.scatter(y, residuals)
plt.xlabel("Critical Temperature (Kelvin)")
plt.ylabel("Residual magnitude (Kelvin)")
plt.title("Distribution of residuals")
plt.show()
