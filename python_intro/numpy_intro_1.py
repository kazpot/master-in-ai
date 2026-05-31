import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(x)
print(type(x))
print(x.dtype)
print(x.shape)

Y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(Y)
print(Y.shape)
print(Y.size)

X = np.array(["Hello", "World"])
print(X)
print("shape:", X.shape)
print("type:", type(x))
print("dtype", X.dtype)

X = np.array([1, 2.5, 4])
print(X, X.dtype)

X = np.array([1.5, 2.5, 3.7], dtype=np.int64)
print(X, X.dtype)

X = np.array([1, 2, 3, 4, 5])
np.save("my_array", X)

Y = np.load("my_array.npy")
print(Y)