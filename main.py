from TimeSeries import *
from TimeSeriesUtils import *
from matplotlib import pyplot as plt
from utils import *
from sklearn.preprocessing import StandardScaler

# CONSTANTS
MOVING_AVG_STEP = 50
WINDOW_TYPE = 'sliding'
WINDOW_SIZE = 2000
WINDOW_STRIDE = 1000
PCA_COMPONENTS = 20