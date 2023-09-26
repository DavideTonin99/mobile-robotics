from models.TimeSeriesUtils import *
from utils.utils import *


class TimeSeries:

    def __init__(self, name, data=None, auto_load=False) -> None:
        """
        **************************************************
        * Class TimeSeries                               *
        **************************************************
        :param name: name of the time series
        :param data: data of the time series (if None, load time series from features csv files)
        :param auto_load: if True, load time series from features csv files
        Exception: data not specified
        """
        self.name = name
        self.window_type = None
        self.window_size = None
        self.window_stride = None

        if data is None and not auto_load:
            raise Exception("Data not specified and auto_load is False")

        if auto_load:
            data = TimeSeriesUtils.ts_load_from_features_csv(
                name, DATA_BASE_PATH)
        self.data = data

    def set_data(self, data) -> None:
        """
        Set the data of the time series
        """
        self.data = data

    def moving_average(self, step) -> None:
        """
        Apply moving average to the time series
        :param step: step of the moving average
        """
        self.data = TimeSeriesUtils.ts_moving_average(self.data, step)

    def prepare_for_window(self, window_type: str, window_size: int, window_stride: int) -> None:
        """
        Prepare the time series for a window
        @param window_type: type of window to add ("sliding" | "tumbling")
        @param window_size: size of the window
        @param window_stride: stride of the window
        """
        self.window_type = window_type
        self.window_size = window_size
        self.window_stride = window_stride

        self.add_window(window_type=window_type, window_size=window_size, window_stride=window_stride)

        step = self.window_stride
        if step == 'tumbling':
            step = self.window_size
        self.remove_window(step=step)

    def add_window(self, window_type="sliding", window_size=None, window_stride=1) -> None:
        """
        Add a window to the time series
        :param window_type: type of window to add ("sliding" | "tumbling")
        :param window_size: size of the window
        :param window_stride: stride of the window
        :raise Exception: window type not supported
        """
        if window_type != "sliding" and window_type != "tumbling":
            raise Exception("Window type not supported")
        if window_size is None or window_size <= 0:
            raise Exception("Window size not specified or negative")

        self.window_type = window_type
        self.window_size = window_size
        self.window_stride = window_stride

        while len(self.data) % self.window_size != 0:
            self.data = self.data[:-1]

        if window_type == "sliding":
            self.sliding_window(window_size, window_stride)
        elif window_type == "tumbling":
            self.tumbling_window(window_size)

    def sliding_window(self, window_size=None, window_stride=1) -> None:
        """
        Add a sliding window to the time series
        :param window_size: size of the window
        :param window_stride: stride of the window
        """
        self.window_type = 'sliding'
        self.window_size = window_size
        self.window_stride = window_stride

        self.data = TimeSeriesUtils.ts_sliding_window(
            self.data, window_size, window_stride)

    def tumbling_window(self, window_size=None) -> None:
        """
        Add a tumbling window to the time series
        :param window_size: size of the window
        """
        self.window_type = 'tumbling'
        self.window_size = window_size
        self.window_stride = window_size

        self.data = TimeSeriesUtils.ts_tumbling_window(
            self.data, window_size)

    def remove_window(self, step=1) -> None:
        """
        Remove a window from the time series (sliding or tumbling)
        :param step: step of the window
        """
        self.data = TimeSeriesUtils.ts_remove_window(
            self.data, n_features=len(TimeSeriesUtils.FEATURES), step=step)

    def normalize(self, normalizer) -> None:
        """
        Normalize the time series
        :param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils.ts_normalize(self.data, normalizer)

    def normalize_inverse(self, normalizer) -> None:
        """
        Inverse normalization of the dataset processed
        :param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils.ts_normalize_inverse(self.data, normalizer)

    def pca(self, model) -> None:
        """
        Apply PCA to the time series
        :param model: PCA model to use
        """
        self.data = TimeSeriesUtils.ts_pca(self.data, model)

    def pca_inverse(self, model) -> None:
        """
        Inverse PCA to the time series
        :param model: PCA model to use
        """
        self.data = TimeSeriesUtils.ts_pca_inverse(self.data, model)
