import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')
print(fuel_econ.shape)
print(fuel_econ.head(10))

plt.scatter(data = fuel_econ, x = "displ", y = "comb")
plt.xlabel("Displacement (1)")
plt.ylabel("Combined Fuel Eff. (mpg)")
plt.show()

sb.regplot(data = fuel_econ, x = "year", y = "comb", x_jitter=0.04, scatter_kws={"alpha": 1/10})
plt.xlabel("Displacement (1)")
plt.ylabel("Combined Fuel Eff. (mpg)")
plt.show()

bins_x = np.arange(0.6, 7 + 0.3, 0.3)
bins_y = np.arange(12, 58 + 3, 3)
plt.hist2d(data = fuel_econ, x = "displ", y = "comb", cmin=0.5, cmap="viridis_r")
plt.colorbar()
plt.xlabel("Displacement (1)")
plt.ylabel("Combined Fuel Eff. (mpg)")
plt.show()

# violin plot
sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses);
sb.violinplot(data=fuel_econ, x='VClass', y='comb');
plt.show()

