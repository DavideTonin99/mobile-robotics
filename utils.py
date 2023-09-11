import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from sklearn import metrics
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sn

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

# NEURAL NETWORK
def print_plot_loss(loss_log, cutStart=0):
    plt.figure(figsize=(20, 5))
    for n, d in loss_log.items():
        plt.plot(d[cutStart:], label=n)
    plt.legend()
    plt.ylabel('')
    plt.show()


def get_loss(x, y, model, loss_function):
    p = model(x)
    loss = loss_function(p, y)
    return loss.item()


def fit(model, x_training, y_training, x_eval, y_eval, optimizer, scheduler, loss_function, epoch=10, print_epoch=True):
    loss_log = {'t': [], 'e': []}
    if print_epoch:
        print('Epoch', 'loss training', 'loss evaluation', sep='\t\t')
    for i in range(epoch):
        model.train()
        p = model(x_training)
        loss = loss_function(p, y_training)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            if i % 50 == 0 and print_epoch:
                lossT = get_loss(x_training, y_training, model, loss_function)
                lossE = get_loss(x_eval, y_eval, model, loss_function)
                loss_log['t'].append(lossT)
                loss_log['e'].append(lossE)
                print(str(i)+'\t', round(lossT, 10), round(lossE, 10), sep='\t')
    return loss_log


def set_grid_square(axs, x_data, y_data):
    '''
    set squared grid
    set limits

    :param axs:
    :return:
    '''
    # Set major and minor grid lines on X
    axs.xaxis.set_major_locator(mticker.MultipleLocator(base=10))
    axs.xaxis.set_minor_locator(mticker.MultipleLocator(base=0.1))

    axs.yaxis.set_major_locator(mticker.MultipleLocator(base=10))
    axs.yaxis.set_minor_locator(mticker.MultipleLocator(base=0.1))

    axs.grid(ls='-', color='black', linewidth=0.8)
    axs.grid(which="minor", ls=':', color='grey', linewidth=0.6)

    # Set limits
    ylim_max, ylim_min = max(y_data), min(y_data)
    xlim_max, xlim_min = max(x_data), min(x_data)
    axs.axis([xlim_min-0.5, xlim_max+0.5, ylim_min-0.5, ylim_max+0.5])
    plt.gca().set_aspect('equal', adjustable='box')


def plot_arrows(timeseries_1, timeseries_2):
    '''
    come plot_scatter però plotta freccie invece che punti
    l'orientamento della freccia viene dal parametro theta
    :param timeseries_1:
    :param timeseries_2:
    :return:
    '''
    x_1 = timeseries_1.iloc[:, 0]
    y_1 = timeseries_1.iloc[:, 1]
    theta_1_cos = np.cos(np.array(timeseries_1.iloc[:, 2]))
    theta_1_sin = np.sin(np.array(timeseries_1.iloc[:, 2]))

    x_2 = timeseries_2.iloc[:, 0]
    y_2 = timeseries_2.iloc[:, 1]
    theta_2_cos = np.cos(np.array(timeseries_2.iloc[:, 2]))
    theta_2_sin = np.sin(np.array(timeseries_2.iloc[:, 2]))

    fig, axs = plt.subplots()
    axs.set_title("banana")

    axs.quiver(x_1, y_1, theta_1_cos, theta_1_sin, width=0.001, color="blue")
    axs.quiver(x_2, y_2, theta_2_cos, theta_2_sin, width=0.001, color="red")

    set_grid_square(axs, x_1, y_1)
    plt.grid()
    plt.show()


def plot_scatter(timeseries_1=None, timeseries_2=None):
    '''
    plotta x, y, theta in 2D plot
    :param timeseries_1: array dati (n righe, 6 colonne)
    :param timeseries_2: array dati (n righe, 6 colonne)
    :return: none
    '''

    x_1 = timeseries_1[:, 0]
    y_1 = timeseries_1[:, 1]
    theta_1 = timeseries_1[:, 2]

    fig, axs = plt.subplots()
    axs.set_title("banana")
    axs.plot(x_1, y_1, color="blue")
    if timeseries_2 is not None:
        x_2 = timeseries_2[:, 0]
        y_2 = timeseries_2[:, 1]
        theta_2 = timeseries_2[:, 2]
        axs.plot(x_2, y_2, color="red")

    set_grid_square(axs, x_1, y_1)
    plt.grid()
    plt.show()

def test_sklearn(y_true, y_pred):
  accuracy = metrics.accuracy_score(y_true, y_pred)
  recall = metrics.recall_score(y_true, y_pred, pos_label=-1)
  precision = metrics.precision_score(y_true, y_pred, pos_label=-1)
  f1 = metrics.f1_score(y_true, y_pred, pos_label=-1)
  roc_auc = metrics.roc_auc_score(y_true, y_pred)

  print('Precision: %f' % precision)
  print('Recall: %f' % recall)
  print('F1 score: %f' % f1)
  print('Accuracy: %f' % accuracy)
  print('Roc auc score: %f' % roc_auc)
  print('%f\t%f\t%f\t%f' % (precision,recall,f1,accuracy))
  print(metrics.precision_recall_fscore_support(y_true, y_pred))

  #index = ['normal', 'attack']
  index = ['attack','normal']
  df_confusion = pd.DataFrame(metrics.confusion_matrix(y_true, y_pred), index=index, columns=index)
  _, ax = plt.subplots()
  sn.heatmap(df_confusion, cmap='Blues', annot=True, fmt='g', ax=ax)
  ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
  ax.set_title('Confusion Matrix')
  #fix bug in lib, info: https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
  #bottom, top = ax.get_ylim()
  #ax.set_ylim(bottom + 0.5, top - 0.5)
  plt.show()

  fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
  plt.figure(1)
  plt.plot(fpr, tpr, label=('ROC curve (area = %0.2f)'%roc_auc))
  plt.plot([0,1],[0,1],'r--')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.show()

  precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(y_true, y_pred)
  plt.figure(1)
  plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=1)
  plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=1)
  plt.title('Precision and recall for different threshold values')
  plt.xlabel('Threshold')
  plt.ylabel('Precision/Recall')
  plt.legend()
  plt.show()
