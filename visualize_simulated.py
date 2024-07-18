from sklearn.datasets import make_classification
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a simulated dataset
sim_data, sim_target = make_classification(n_samples=500, n_features=10, n_informative=6, n_classes=3,
                                           class_sep=2.0, n_clusters_per_class=2, random_state=42)

sim_df = pd.DataFrame(data=sim_data, columns=[f'feature_{i}' for i in range(10)])

# Scatter Plot
plt.figure(figsize=(4, 4))
sns.scatterplot(x=sim_df['feature_0'], y=sim_df['feature_1'], hue=sim_target, palette='viridis')
#plt.title('Scatter Plot of Feature 0 vs Feature 1')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(title='Class')
plt.savefig('sim_scatter.jpg', dpi = 600, bbox_inches = 'tight')
plt.show()


# Heatmap of feature correlations
plt.figure(figsize=(8, 7))
sns.heatmap(sim_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
#plt.title('Heatmap of Feature Correlations in Simulated Dataset')
plt.savefig('sim_heat.jpg', dpi = 600, bbox_inches = 'tight')
plt.show()
