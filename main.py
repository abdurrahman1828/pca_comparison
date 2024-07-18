import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, IncrementalPCA, KernelPCA
from sklearn.datasets import load_wine, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
# Set seed for reproducibility
np.random.seed(42)

# Load the Wine dataset
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_target = wine.target

# Create a simulated dataset
sim_data, sim_target = make_classification(n_samples=500, n_features=10, n_informative=6, n_classes=3,
                                           class_sep=2.0, n_clusters_per_class=2,random_state=42)


# Standardize the data
scaler = StandardScaler()
wine_scaled = scaler.fit_transform(wine_df)
sim_scaled = scaler.fit_transform(sim_data)


def perform_pca_analysis(data, n_components=10, method='PCA'):
    if method == 'PCA':
        model = PCA(n_components=n_components)
    elif method == 'SparsePCA':
        model = SparsePCA(n_components=n_components, random_state=42)
    elif method == 'IncrementalPCA':
        model = IncrementalPCA(n_components=n_components)
    elif method == 'KernelPCA':
        model = KernelPCA(n_components=n_components, kernel='rbf')
    else:
        raise ValueError("Method not recognized.")

    start_time = time.time()
    principal_components = model.fit_transform(data)
    end_time = time.time()

    if method in ['PCA', 'IncrementalPCA']:
        explained_variance = model.explained_variance_ratio_
    else:
        explained_variance = None  # SparsePCA and KernelPCA do not have explained variance
    print(f"{method} computation time:", end_time - start_time, "seconds")
    return model, principal_components, explained_variance


def plot_pca_results(principal_components, target, title):
    plt.figure(figsize=(4, 3))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=target, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.savefig(f'{title}.jpg', dpi = 600, bbox_inches = 'tight')
    plt.show()


datasets = {
    'Wine': (wine_scaled, wine_target),
    'Simulated': (sim_scaled, sim_target)
}

methods = ['PCA', 'SparsePCA', 'IncrementalPCA', 'KernelPCA']

results = []

for dataset_name, (data, target) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

    for method in methods:
        model, principal_components, explained_variance = perform_pca_analysis(X_train, method=method)
        X_train_pca = model.transform(X_train)
        X_test_pca = model.transform(X_test)

        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_pca, y_train)
        y_pred = svm.predict(X_test_pca)

        accuracy = accuracy_score(y_test, y_pred)
        results.append((dataset_name, method, accuracy))

        plot_title = f'{method} on {dataset_name} Data'
        plot_pca_results(model.transform(data), target, plot_title)

        if explained_variance is not None:
            print(f'Explained variance for {method} on {dataset_name} Data: {explained_variance}')
        print(f'Accuracy for {method} on {dataset_name} Data: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))
