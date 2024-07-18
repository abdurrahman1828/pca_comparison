import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Set seed for reproducibility
np.random.seed(42)

# Load the Wine dataset
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_target = wine.target

# Scatter Plot
plt.figure(figsize=(4, 4))
sns.scatterplot(x=wine_df['alcohol'], y=wine_df['malic_acid'], hue=wine_target, palette='viridis')
#plt.title('Scatter Plot of Alcohol vs Malic Acid')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.legend(title='Cultivar')
plt.savefig('wines_scatter.jpg', dpi = 600, bbox_inches = 'tight')
plt.show()

# Heatmap of feature correlations
plt.figure(figsize=(8, 7))
sns.heatmap(wine_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
#plt.title('Heatmap of Feature Correlations in Wine Dataset')
plt.savefig('wines_heat.jpg', dpi = 600, bbox_inches = 'tight')
plt.show()
