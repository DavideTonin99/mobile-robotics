from TimeSeries import *
from TimeSeriesUtils import *
from Dataset import *
from matplotlib import pyplot as plt
from utils import *
from sklearn.preprocessing import StandardScaler

# CONSTANTS
MOVING_AVG_STEP = 50
WINDOW_TYPE = 'sliding'
WINDOW_SIZE = 20
WINDOW_STRIDE = 5
PCA_COMPONENTS = 10

scaler = StandardScaler()

dataset = {}

# Load the time series for training
# TRAINING
for i in range(1, 8):
    t = TimeSeries(f'nominal_{i}', auto_load=True)
    dataset[t.name] = t

t_train_no_window = TimeSeries('train no window', data=np.concatenate([
                               t.data for t in dataset.values()]))
dataset_train_no_window = Dataset('train_no_window', data=dataset)

# time series without window
t_original = t_train_no_window.data

for t_name, t in dataset.items():
    t.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
    dataset[t.name] = t

dataset_train = Dataset('train', data=dataset)

t_train = TimeSeries('train', data=np.concatenate(
    [t.data for t in dataset_train.data.values()]))

# Fit the scaler model
scaler = StandardScaler()
scaler.fit(t_train.data)

t_train.normalize(scaler)
model = fit_pca(t_train.data, n_components=PCA_COMPONENTS,
                show_plot_variance=False)

# END TRAINING

# PCA ON TRAINING DATASET
t_train.pca(model)
t_train.pca_inverse(model)
t_train.normalize_inverse(scaler)
t_train.remove_window(step=WINDOW_STRIDE)

t_predict = t_train.data

plot_ts("Figure Test", ts={'original': t_original, 'predict': t_predict}, features=TimeSeriesUtils.PLOT_FEATURES,
        n_rows=2, n_cols=3, figsize=(15, 5), colors={'original': 'black', 'predict': 'orange'}, ts_name='train', save_plot=True)

# LOAD TEST DATASET
dataset_test_original = {}
for i in range(1, 11):
    t = TimeSeries(f'anomaly_{i}', auto_load=True)
    dataset_test_original[t.name] = t

dataset_test = {}
for t_name, t in dataset_test_original.items():
    t.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
    t.normalize(scaler)
    t.pca(model)
    t.pca_inverse(model)
    t.normalize_inverse(scaler)
    t.remove_window(step=WINDOW_STRIDE)
    plot_ts(f'Figure {t_name}', ts={'original': dataset_test_original[t_name].data, 'predict': t.data}, features=TimeSeriesUtils.PLOT_FEATURES,
        n_rows=2, n_cols=3, figsize=(15, 5), colors={'original': 'black', 'predict': 'orange'}, ts_name=t_name, save_plot=True)