# pca_comparison
This code includes the comparison of the PCA method and its variants on simulated and real-world (wine) data.

I generated simulated data using the `sklearn` library. The `wine` data is available in the UCI ML repository. The representations of the PCA models were used as input to an SVM classifier. 

The evaluation pipeline:

![plot](assets/pipeline.jpg)

Principle components for **simulated** data:
1. PCA:
![plot](assets/PCA%20on%20Simulated%20Data.jpg)
2. Sparse PCA:
![plot](assets/SparsePCA%20on%20Simulated%20Data.jpg)
3. Incremental PCA:
![plot](assets/IncrementalPCA%20on%20Simulated%20Data.jpg)
4. Kernel PCA:
![plot](assets/KernelPCA%20on%20Simulated%20Data.jpg)

Principle components for **wine** data:
1. PCA:
![plot](assets/PCA%20on%20Wine%20Data.jpg)
2. Sparse PCA:
![plot](assets/SparsePCA%20on%20Wine%20Data.jpg)
3. Incremental PCA:
![plot](assets/IncrementalPCA%20on%20Wine%20Data.jpg)
4. Kernel PCA:
![plot](assets/KernelPCA%20on%20Wine%20Data.jpg)


Classification Results:
![plot](assets/results.png)