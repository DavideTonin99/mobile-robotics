from matplotlib import pyplot as plt
from utils import *
from sklearn.preprocessing import StandardScaler
from NNLinearRegression import *
from RegressionTCN import *
import torch
from torch import nn
import copy
from sklearn.neighbors import LocalOutlierFactor

from TimeSeries import *
from TimeSeriesUtils import *
from Dataset import *
from utils import *
from plot_time_series import *

# CONSTANTS
WINDOW_TYPE = 'sliding'
WINDOW_SIZE = 30
WINDOW_STRIDE = 30
COLUMNS = 6

scaler_model = StandardScaler()
dataset = {}

device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# LOAD TRAINING DATA
dataset = {}
t_list = [f'nominal_{i}' for i in range(7, 8)]
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    dataset[t_name] = t

# train dataset
dataset_train = Dataset(f'train {WINDOW_TYPE} window ws={WINDOW_SIZE}, stride={WINDOW_STRIDE}', time_series=dataset)

dataset_train_process = copy.deepcopy(dataset_train)
dataset_train_process.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
scaler_model.fit(dataset_train_process.ts_data.data)
dataset_train_process.normalize(normalizer=scaler_model, mode='complete')

# LOAD EVALUATION DATA
dataset = {}
t_list = [f'nominal_{i}' for i in range(6, 7)]
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    dataset[t_name] = t

# train dataset
dataset_eval = Dataset(f'evaluation {WINDOW_TYPE} window ws={WINDOW_SIZE}, stride={WINDOW_STRIDE}', time_series=dataset)

dataset_eval_process = copy.deepcopy(dataset_eval)
dataset_eval_process.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
dataset_eval_process.normalize(normalizer=scaler_model, mode='complete')

# model = NNLinearRegression(COLUMNS*WINDOW_SIZE, COLUMNS*WINDOW_SIZE)
# model = TCN(COLUMNS*WINDOW_SIZE, COLUMNS*WINDOW_SIZE, [1]*3, kernel_size=2, dropout=0.2)
model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
model.fit(dataset_train_process.ts_data.data)
y_pred = model.predict(dataset_eval_process.ts_data.data)
test_sklearn(dataset_eval_process.ts_data.data, y_pred)

# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.0001)
# loss_function = nn.MSELoss()
# scheduler = None

# x_training = torch.unsqueeze(torch.from_numpy(dataset_train_process.ts_data.data).float(), 0).to(device)
# y_training = torch.unsqueeze(torch.from_numpy(dataset_train_process.ts_data.data).float(), 0).to(device)
# x_eval = torch.unsqueeze(torch.from_numpy(dataset_eval_process.ts_data.data).float(), 0).to(device)
# y_eval = torch.unsqueeze(torch.from_numpy(dataset_eval_process.ts_data.data).float(), 0).to(device)
# model = model.to(device)

# loss_log = fit(model=model, x_training=x_training, y_training=y_training, x_eval=x_eval, y_eval=y_eval, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, epoch=20000, print_epoch=True)

# q1, q2 = np.quantile(loss_log['t'], [0.05, 0.95], axis=0)
# pc_iqr = q2 - q1
# pc_lower_bound = q1 - (1 * pc_iqr)
# pc_upper_bound = q2 + (1 * pc_iqr)

# print(f"* Lower Bound: {pc_lower_bound}")
# print(f"* Upper Bound: {pc_upper_bound}")

# print_plot_loss(loss_log, cutStart=10)

# LOAD TEST DATASET
t_list = [f'anomaly_{i}' for i in range(1, 11)]
test = {}
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    test[t_name] = t

t_list = [f'nominal_{i}' for i in range(1, 6)]
for t_name in t_list:
    t = TimeSeries(t_name, auto_load=True)
    test[t_name] = t

# test dataset
dataset_test = Dataset(f'test {WINDOW_TYPE} window ws={WINDOW_SIZE}, stride={WINDOW_STRIDE}', time_series=test)

dataset_test_process = copy.deepcopy(dataset_test)
dataset_test_process.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
dataset_test_process.normalize(normalizer=scaler_model, mode='single')

for t_name, t in dataset_test_process.time_series.items():
    is_anomaly_gt = 'anomaly' in t_name
    is_anomaly_pred = False
    count_windows_anomaly = 0
    errors_list = []

    for i, row_window in enumerate(t.data):
        row_window = np.array([row_window])
        x = torch.from_numpy(row_window).float().to(device)
        y = torch.from_numpy(row_window).float().to(device)
        model.eval()
        prediction = model(x)
        prediction = prediction.cpu().detach().numpy()
        error = (np.square(prediction - row_window).mean(axis=1))
        anomalies_mask = ((error > pc_upper_bound) | (error < pc_lower_bound)) == True
        if anomalies_mask.any():
            count_windows_anomaly += 1
            is_anomaly_pred = True
            errors_list.append(error)
        
    print(f"{t_name}: Anomalies detected: {count_windows_anomaly}, {errors_list}")