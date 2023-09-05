import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATA_BASE_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'dataset')
IMAGES_BASE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'images')


def plot_ts(title, ts, features, n_rows, n_cols, figsize=(15, 5), colors={}) -> None:
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title)
    row = 0
    col = 0
    feature_index = 0

    for _, cols in features.items():
        col = 0
        for col_name in cols:
            for key, time_series in ts.items():
                axs[row, col].plot(time_series[:, feature_index], color=colors[key])
                axs[row, col].set_title(f'{col_name}')
                axs[row, col].yaxis.set_ticks(np.arange(float(time_series[:, feature_index].min()), float(time_series[:, feature_index].max()), 0.05))
            col += 1
            feature_index += 1
        row += 1

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
