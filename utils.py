import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
IMAGES_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')

def fit_pca(dataset, n_components=21, show_plot_variance=False) -> PCA:
    """
    Fit a PCA model to the dataset
    :param dataset: dataset to fit the PCA model
    :param n_components: number of components to keep
    :param show_plot_variance: show the plot of the variance
    """
    pca = PCA(n_components=n_components)
    pca.fit(dataset)
    if show_plot_variance:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
    return pca
