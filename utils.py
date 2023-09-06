import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATA_BASE_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'dataset')
IMAGES_BASE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'images')


def plot_ts(title, ts, features, n_rows, n_cols, figsize=(15, 5), colors={}, ts_name=None, show_plot=False, save_plot=False) -> None:
    """
    Plot the time series
    :param title: title of the plot
    :param ts: time series to plot
    :param features: features to plot
    :param n_rows: number of rows of the plot
    :param n_cols: number of columns of the plot
    :param figsize: size of the plot
    :param colors: colors of the plot
    :param ts_name: name of the time series
    :param show_plot: show the plot
    :param save_plot: save the plot
    :return: None
    """
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
                # axs[row, col].yaxis.set_ticks(np.arange(float(time_series[:, feature_index].min()), float(time_series[:, feature_index].max()), 0.05))
            col += 1
            feature_index += 1
        row += 1

    if save_plot:
        if not os.path.isdir(IMAGES_BASE_PATH):
            os.mkdir(IMAGES_BASE_PATH)

        image_path = os.path.join(IMAGES_BASE_PATH, f"{ts_name}_pca.png")
        fig.savefig(image_path)

        plt.cla()

    if show_plot:
        plt.show()

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
