import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import os

# matplotlib.use('TkAgg')

FILEPATH_1 = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'dataset', 'nominal_7', 'ml_data_nominal_3.csv')
# FILEPATH_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'nominal_2', 'ml_data_nominal_2.csv')
# FILEPATH_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'nominal_3', 'ml_data_nominal_3.csv')
# FILEPATH_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'nominal_4', 'ml_data_nominal_4.csv')
# FILEPATH_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'nominal_5', 'ml_data_nominal_5.csv')
# FILEPATH_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'nominal_6', 'ml_data_nominal_6.csv')
# FILEPATH_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'nominal_7', 'ml_data_nominal_7.csv')

FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'dataset', 'anomaly_1', 'ml_data_anomaly_1.csv')
FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'dataset', 'anomaly_2', 'ml_data_anomaly_2.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_3', 'ml_data_anomaly_3.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_4', 'ml_data_anomaly_4.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_5', 'ml_data_anomaly_5.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_6', 'ml_data_anomaly_6.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_7', 'ml_data_anomaly_7.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_8', 'ml_data_anomaly_8.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_9', 'ml_data_anomaly_9.csv')
# FILEPATH_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'anomaly_10', 'ml_data_anomaly_10.csv')

FILEPATH = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'dataset', 'anomaly_2', 'ml_data_anomaly_2.csv')


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


if __name__ == "__main__":
    csv_data_1 = pd.read_csv(FILEPATH_1, sep=",")
    csv_data_2 = pd.read_csv(FILEPATH_2, sep=",")

    # csv_data = pd.read_csv(FILEPATH, sep=",")
    plot_arrows(csv_data_1, csv_data_2)
    # plot_scatter(csv_data)
    # plot_scatter(csv_data_1, csv_data_2)
