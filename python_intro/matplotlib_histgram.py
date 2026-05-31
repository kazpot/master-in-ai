import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
pokemon.head(10)

bins = np.arange(0, pokemon["speed"].max()+5, 5)
plt.hist(data = pokemon, x = "speed", bins = bins)
plt.show()

bins = np.arange(0, pokemon["height"].max()+0.2, 0.2)
plt.hist(data = pokemon, x = "height", bins = bins)
plt.xlim((0, 6))
plt.show()

print(np.log10(pokemon["weight"].describe()))

bins = 10 ** np.arange(-1, 3 + 0.1, 0.1)
ticks = [0.1, 0.3, 1, 10, 30, 100, 300, 1000]
labels = ["{}".format(v) for v in ticks]
plt.hist(data = pokemon, x = "weight", bins = bins)
plt.xscale("log")
plt.xticks(ticks, labels)
plt.show()

