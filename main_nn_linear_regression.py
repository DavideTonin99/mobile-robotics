from TimeSeries import *
from TimeSeriesUtils import *
from Dataset import *
from matplotlib import pyplot as plt
from utils import *
from sklearn.preprocessing import StandardScaler
from plot_time_series import *
from NNLinearRegression import *
import torch
from torch import nn

def printPlotLoss(lossLog, cutStart=0):
    plt.figure(figsize=(20, 5))
    for n, d in lossLog.items():
        plt.plot(d[cutStart:], label=n)
    plt.legend()
    plt.ylabel('')
    plt.show()


def getLoss(x, y, model, lossFun):
    p = model(x)   # rnn output
    loss = lossFun(p, y)    # calculate loss
    return loss.item()


def fit(model, xT, yT, xE, yE, scheduler, lossFun, epoch=10, printEpoch=True):
    lossLog = {'t': [], 'e': []}
    if printEpoch:
        print('Epoch', 'lossL', sep='\t\t')
    for i in range(epoch):
        model.train()
        p = model(xT)
        loss = lossFun(p, yT)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            if i % 50 == 0 and printEpoch:
                lossT = getLoss(xT, yT, model, lossFun)
                lossE = getLoss(xE, yE, model, lossFun)
                lossLog['t'].append(lossT)
                lossLog['e'].append(lossE)
                print(str(i)+'\t', round(lossT, 10), round(lossE, 10), sep='\t')
    return lossLog


# CONSTANTS
MOVING_AVG_STEP = 50
WINDOW_TYPE = 'sliding'
WINDOW_SIZE = 30
WINDOW_STRIDE = 30
# PCA_COMPONENTS = 10

scaler = StandardScaler()
dataset = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the time series for training
# TRAINING
for i in range(7, 8):
    t = TimeSeries(f'nominal_{i}', auto_load=True)
    dataset[t.name] = t

t_train_no_window = TimeSeries('train no window', data=np.concatenate([
                               t.data for t in dataset.values()]))
dataset_train_no_window = Dataset('train_no_window', data=dataset)

# time series without window
t_original = t_train_no_window.data

for t_name, t in dataset.items():
    t.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE,
                 stride=WINDOW_STRIDE)
    dataset[t.name] = t

dataset_train = Dataset('train', data=dataset)

t_train = TimeSeries('train', data=np.concatenate(
    [t.data for t in dataset_train.data.values()]))

# evaluation
dataset = {}
for i in range(2, 3):
    t = TimeSeries(f'nominal_{i}', auto_load=True)
    dataset[t.name] = t

t_eval_no_window = TimeSeries('eval no window', data=np.concatenate([
                               t.data for t in dataset.values()]))
dataset_eval_no_window = Dataset('eval_no_window', data=dataset)

# time series without window
t_original = t_eval_no_window.data

for t_name, t in dataset.items():
    t.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE,
                 stride=WINDOW_STRIDE)
    dataset[t.name] = t

dataset_eval = Dataset('eval', data=dataset)

t_eval = TimeSeries('eval', data=np.concatenate(
    [t.data for t in dataset_eval.data.values()]))

# Fit the scaler model
scaler = StandardScaler()
scaler.fit(t_train.data)
t_train.normalize(scaler)
t_eval.normalize(scaler)

model = NNLinearRegression(t_train.data.shape[1], t_train.data.shape[1])

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
lossFun = nn.MSELoss()
scheduler = None

xT = torch.from_numpy(t_train.data).float().to(device)
yT = torch.from_numpy(t_train.data).float().to(device)
xE = torch.from_numpy(t_eval.data).float().to(device)
yE = torch.from_numpy(t_eval.data).float().to(device)
model = model.to(device)

lossLog = fit(model=model, xT=xT, yT=yT, xE=xE, yE=yE, scheduler=scheduler, lossFun=lossFun, epoch=10000, printEpoch=True)

# calcolare il percentile per avere la threshold
# il percentile lo calcolo da lossLog.e che contiene gli errori di predizione per la traiettoria di evaluation

# printPlotLoss(lossLog, cutStart=10)

# LOAD TEST DATASET
dataset_test_original = {}
for i in range(1, 11):
    t = TimeSeries(f'anomaly_{i}', auto_load=True)
    dataset_test_original[t.name] = t

dataset_test = {}
for t_name, t in dataset_test_original.items():
    t.add_window(type=WINDOW_TYPE, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
    t.normalize(scaler)

    for row in t.data:
        row = np.array([row])
        x = torch.from_numpy(row).float().to(device)
        y = torch.from_numpy(row).float().to(device)
        model.eval()
        result = model(x)
        result = result.cpu().detach().numpy()
        result = (np.square(result - row).mean(axis=1))
        print(result)
        # result = errore di predizione della riga (ovvero finestra) corrente
        # mettere result in un array
    # confrontare l'array con la soglia per identificare le anomalie per la traiettoria corrente
