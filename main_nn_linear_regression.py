from sklearn.preprocessing import StandardScaler
from main import *

from Dataset import *
from utils import *

# CONSTANTS
WINDOW_TYPE = 'sliding'
WINDOW_SIZE = 30
WINDOW_STRIDE = 15
COLUMNS = 6

scaler_model = StandardScaler()

device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

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

model = NNLinearRegression(6 * WINDOW_SIZE, 6 * WINDOW_SIZE)

# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.0001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
scheduler = None

main_nn_linear_regression(dataset_train=dataset_train, model=model, scaler_model=scaler_model, window_type=WINDOW_TYPE,
                          window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE, device=device, optimizer=optimizer,
                          loss_function=loss_function, epoch=25000, scheduler=scheduler, dataset_eval=dataset_eval,
                          dataset_test=dataset_test)
