from TimeSeries import *
from TimeSeriesUtils import *
from Dataset import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from plot_time_series import *
from main import *
from utils import *

# CONSTANTS
MOVING_AVG_STEP = 50
WINDOW_TYPE = 'sliding'
WINDOW_SIZE = 70
WINDOW_STRIDE = 30
PCA_COMPONENTS = 7

scaler_model = StandardScaler()

# training dataset
dataset = {}
t_list = [f'nominal_{i}' for i in range(7, 8)]
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    dataset[t_name] = t

dataset_train = Dataset(
    f'train {WINDOW_TYPE} window ws={WINDOW_SIZE}, stride={WINDOW_STRIDE}', time_series=dataset)

# evaluation dataset
dataset = {}
t_list = [f'nominal_{i}' for i in range(1, 7)]
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    dataset[t_name] = t

dataset_eval = Dataset(
    f'evaluation {WINDOW_TYPE} window ws={WINDOW_SIZE}, stride={WINDOW_STRIDE}', time_series=dataset)

# test dataset
t_list = [f'anomaly_{i}' for i in range(1, 11)]
test = {}
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    test[t_name] = t

t_list = [f'nominal_{i}' for i in range(1, 7)]
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    test[t_name] = t

dataset_test = Dataset(
    f'test {WINDOW_TYPE} window ws={WINDOW_SIZE}, stride={WINDOW_STRIDE}', time_series=test)

main_pca(dataset_train=dataset_train, scaler_model=scaler_model, window_type=WINDOW_TYPE,
         window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE, pca_components=PCA_COMPONENTS, show_plot_variance=False, threshold_method='local_outlier_factor', dataset_eval=dataset_eval, dataset_test=dataset_test)
