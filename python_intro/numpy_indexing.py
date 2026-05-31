import numpy as np

X = np.arange(25).reshape(5, 5)
print(X)

print(X[X > 10])
print(X[(X > 10) & (X < 17)])
X[(X > 10) & (X < 17)] = -1
print(X)

x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 2, 8, 4])

print(np.intersect1d(x, y))
print(np.setdiff1d(x, y))
print(np.union1d(x, y))

x = np.random.randint(1, 11, size=(10, ))
print(x)
print(np.sort(x))
print(x)

x.sort()
print(x)

X = np.random.randint(1, 11, size=(5, 5))
print(X)

print(np.sort(X, axis=0))

print(np.sort(X, axis=1))

# TODO: replace None with appropriate code
# Create a 5 x 5 ndarray with consecutive integers from 1 to 25 (inclusive).
X = np.arange(1, 26).reshape(5, 5)
print(X)

# TODO: replace None with appropriate code
# Use Boolean indexing to pick out only the odd numbers in the array
Y = X[X % 2 != 0]
print(Y)