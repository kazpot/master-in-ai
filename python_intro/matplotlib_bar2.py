import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv("pokemon.csv")
print(pokemon.shape)
print(pokemon.head())

pkmn_types = pokemon.melt(id_vars=['id', 'species'], 
                          value_vars=['type_1', 'type_2'], 
                          var_name='type_level', 
                          value_name='type')

print(pkmn_types[802:812])

type_counts = pkmn_types["type"].value_counts()
type_order = type_counts.index
base_color = sb.color_palette()[0]

n_pokemon = pokemon.shape[0]
max_type_count = type_counts[0]
max_prop = max_type_count / n_pokemon
print(max_prop)

tick_props = np.arange(0, max_prop, 0.02)
tick_names = ["{:0.2f}".format(v) for v in tick_props]

sb.countplot(data = pkmn_types, y = "type", color = base_color, order = type_order)
plt.xticks(tick_props * n_pokemon, tick_names)
plt.xlabel("proportion")
plt.show()