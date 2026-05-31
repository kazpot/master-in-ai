import numpy as np
import matplotlib.pyplot as plt

# define vector v
v = np.array([1, 1])

ax = plt.axes()
ax.plot(0, 0, "or")
ax.arrow(0, 0, *v, color="b", linewidth=2.0, head_width=0.2, head_length=0.25)

# Define scalar a
a = 3
av = 3 * v
ax.arrow(0, 0, *av, color='c', linestyle="dotted", linewidth=2.5, head_width=0.30, head_length=0.35)

plt.xlim(-2, 4)
major_xticks = np.arange(-2, 5)
ax.set_xticks(major_xticks)

plt.ylim(-1, 4)
major_yticks = np.arange(-1, 5)
ax.set_yticks(major_yticks)

plt.grid(visible=True, which='major')
plt.show()