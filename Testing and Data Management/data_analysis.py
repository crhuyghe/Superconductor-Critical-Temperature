# Created by Christian Huyghe
# Uses multiple strategies to visualize the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
# from ucimlrepo import fetch_ucirepo

# fetch dataset
# superconductivity_data = fetch_ucirepo(id=464)
#
# # data (as pandas dataframes)
# X = superconductivity_data.data.features
# y = superconductivity_data.data.targets
# X.to_csv("features.csv", index=False)
# y.to_csv("target.csv", index=False)

X = pd.read_csv('../Data/features.csv')
y = pd.read_csv('../Data/target.csv')

norm_X = (X - X.mean(axis=0)) / X.std(axis=0)

corr = [float(abs(np.corrcoef(X.iloc[:, i].to_numpy(), y.to_numpy().T)[0, 1])) for i in range(len(X.keys()))]
corr.sort()
print(corr)

plt.hist(corr)
plt.title("Feature-Target Correlation")
plt.xlabel("Frequency")
plt.ylabel("Correlation")
plt.show()


plt.hist(y)
plt.ylabel('Frequency')
plt.xlabel("Critical Temperature (Kelvin)")
plt.title("Dataset Critical Temperatures")
plt.show()

cov = np.cov(norm_X.T)
eig_val, eig_vec = np.linalg.eig(cov)
eig_val.sort()
print(eig_val[::-1])
mse = np.sum(eig_val[:-22])
print(mse, mse**.5)
plt.plot(eig_val[::-1])
plt.ylabel("Eigenvalue magnitude")
plt.xlabel("Eigenvalue rank")
plt.title("Dataset Eigenvalues by Magnitude")
plt.show()

pca = decomposition.PCA(n_components=2)
transformed_data = pca.fit_transform(norm_X)

plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=y.to_numpy(), cmap='viridis')
plt.title('PCA of data with two components')
plt.colorbar()
plt.show()

tsne = manifold.TSNE(n_components=2, random_state=0)
transformed_data = tsne.fit_transform(norm_X)

plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=y.to_numpy(), cmap='viridis')
plt.title('t-SNE of data with two components')
plt.colorbar()
plt.show()
