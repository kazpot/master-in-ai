import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
print(x)
print(y)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

X = np.array([1, 2, 3, 4]).reshape(2, 2)
print(X)

Y = np.array([5, 6, 7, 8]).reshape(2, 2)
print(Y)

print(X + Y)
print(X - Y)
print(X * Y)
print(X / Y)

print(x)

print(np.sqrt(x))
print(np.exp(x))
print(np.power(x, 2))

print(X)
print("average of all:", X.mean())

print("average of columns:", X.mean(axis=0))
print("average of rows:", X.mean(axis=1))

print(X.std())
print(np.median(X))
print(X.max())
print(X.min())

print(X)
print(3 + X)
print(X - 3)
print(X * 3)
print(X / 3)

Y = np.arange(9).reshape(3, 3)
print(Y)

X = np.arange(3)
print(X)

print(Y + X)

z = np.arange(3).reshape(3, 1)
print(z)

print(z + Y)

X = np.ones((4,4)) * np.arange(1, 5)
print(X)