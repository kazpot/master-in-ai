import numpy as np

X = np.arange(1, 21).reshape(4, 5)
print(X)

z = X[1:4, 2:5]
print(z)

z = X[1:, 2:]
print(z)

z = X[:3, 2:]
print(z)

z = X[:, 2]
print(z)

z = X[:, 2:3]
print(z)

z = X[1:, 2:]
print(z)

z[2, 2] = 555
print(z)
print(X)

X = np.arange(20).reshape(4, 5)
print(X)

z = X[1:, 2:].copy()
print(z)
z[2, 2] = 555
print(z)
print(X)

indices = np.array([1, 3])
print(indices)
y = X[indices, :]
print(y)

z = X[:, indices]
print(z)

z = np.diag(X)
print(z)

z = np.diag(X, k=1)
print(z)

z = np.diag(X, k=-1)
print(z)

X = np.array([[1, 2, 3], [5, 2, 8], [1, 2, 3]])
print(X)
print(np.unique(X))