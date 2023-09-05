from TimeSeries import *
from TimeSeriesUtils import *
from Dataset import *
from matplotlib import pyplot as plt
from utils import *
from sklearn.preprocessing import StandardScaler

# CONSTANTS
MOVING_AVG_STEP = 50
WINDOW_TYPE = 'sliding'
WINDOW_SIZE = 50
WINDOW_STRIDE = 10
PCA_COMPONENTS = 10

scaler = StandardScaler()

dataset = {}

# Load the time series
t1 = TimeSeries('01_0606', auto_load=True)
dataset[t1.name] = t1

# plot_ts("Figure Test", ts={'original': t1}, features=TimeSeriesUtils.PLOT_FEATURES,
#         n_rows=2, n_cols=3, figsize=(15, 5), colors={'original': 'black', 'predict': 'orange'})
# plt.show()
# exit(1)

t2 = TimeSeries('01_0607', auto_load=True)
dataset[t2.name] = t2

# t3 = TimeSeries('02_0606', auto_load=True)
# dataset[t3.name] = t3

t_train_no_window = TimeSeries('train no window', data=np.concatenate([
                               t.data for t in dataset.values()]))
dataset_train_no_window = Dataset('train_no_window', data=dataset)

t1.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
t2.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
# t3.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)

dataset[t1.name] = t1
dataset[t2.name] = t2
# dataset[t3.name] = t3

dataset_train = Dataset('train', data=dataset)

t_train = TimeSeries('train', data=np.concatenate(
    [t.data for t in dataset_train.data.values()]))

t_original = t_train.data
t_original = TimeSeriesUtils._remove_window(t_original, len(
    TimeSeriesUtils.FEATURES), step=WINDOW_STRIDE)

# Fit the scaler model
scaler = StandardScaler()
scaler.fit(t_train.data)

t_train.normalize(scaler)
model = fit_pca(t_train.data, n_components=PCA_COMPONENTS,
                show_plot_variance=False)

t_train.pca(model)
t_train.pca_inverse(model)
t_train.normalize_inverse(scaler)
t_train.remove_window(step=WINDOW_STRIDE)

t_predict = t_train.data

plot_ts("Figure Test", ts={'original': t_original, 'predict': t_predict}, features=TimeSeriesUtils.PLOT_FEATURES,
        n_rows=2, n_cols=3, figsize=(15, 5), colors={'original': 'black', 'predict': 'orange'})
plt.show()
