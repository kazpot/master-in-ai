import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv("pokemon.csv")
print(pokemon.shape)
print(pokemon.head())

base_color = sb.color_palette()[0]
gen_order = pokemon["generation_id"].value_counts().index
sb.countplot(data = pokemon, x = "generation_id", color= base_color, order= gen_order)
plt.show()

base_color = sb.color_palette()[0]
type_order = pokemon["type_1"].value_counts().index
sb.countplot(data = pokemon, y = "type_1", color= base_color, order = type_order)
plt.show()