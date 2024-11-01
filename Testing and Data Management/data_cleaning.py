# Created by Christian Huyghe on 10/29/2024
# Takes in the CSV of superconductor data, normalizes it, and removes outliers.
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

X = pd.read_csv('../Data/features.csv')
y = pd.read_csv('../Data/target.csv')

scaler = StandardScaler()

norm_X = scaler.fit_transform(X)

iforest = IsolationForest(contamination=0.001, random_state=20)
combined_data = np.column_stack((norm_X, y - np.mean(y, axis=0) / np.std(y, axis=0)))
iforest.fit(combined_data)
outliers = iforest.predict(combined_data)
cleaned_X = norm_X[outliers == 1]
cleaned_y = y[outliers == 1]

norm_X = scaler.fit_transform(scaler.inverse_transform(cleaned_X))
mean_X = scaler.mean_
std_X = scaler.scale_

pd.DataFrame(cleaned_X, columns=X.keys(), dtype="float32").to_csv("../Cleaned Data/cleaned_features.csv", index=False)
pd.DataFrame(cleaned_y, columns=y.keys()).to_csv("../Cleaned Data/cleaned_target.csv", index=False)
json.dump({"mean": list(mean_X), "std": list(std_X)}, open("../Cleaned Data/feature_statistics.json", "w"))
