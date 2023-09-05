from TimeSeriesUtils import *
from utils import *


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

        if data is None and not auto_load:
            raise Exception("Data not specified and auto_load is False")

        if auto_load:
            data = TimeSeriesUtils._load_from_features_csv(
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
        self.data = TimeSeriesUtils._moving_average(self.data, step)

    def normalize(self, normalizer) -> None:
        """
        Normalize the time series
        :param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils._normalize(
            self.data, normalizer)

    def add_window(self, type="sliding", window_size=None, stride=1) -> None:
        """
        Add a window to the time series
        :param type: type of window to add ("sliding" | "tumbling")
        :param window_size: size of the window
        Exception: window type not supported
        """
        if window_size is not None:
            self.window_size = window_size

            if type != "sliding" and type != "tumbling":
                raise Exception("Window type not supported")

            while len(self.data) % self.window_size != 0:
                self.data = self.data[:-1]

            if type == "sliding":
                self.sliding_window(window_size, stride)
            elif type == "tumbling":
                self.tumbling_window(window_size)
        else:
            raise Exception("Window size not specified")

    def sliding_window(self, window_size=None, stride=1) -> None:
        """
        Add a sliding window to the time series
        :param window_size: size of the window
        :param stride: stride of the window
        """
        self.data = TimeSeriesUtils._sliding_window(
            self.data, window_size, stride)

    def tumbling_window(self, window_size=None) -> None:
        """
        Add a tumbling window to the time series
        :param window_size: size of the window
        """
        self.data = TimeSeriesUtils._tumbling_window(
            self.data, window_size)

    def remove_window(self, step=1) -> None:
        """
        Remove a window from the time series (sliding or tumbling)
        :param step: step of the window
        """
        self.data = TimeSeriesUtils._remove_window(
            self.data, n_features=len(TimeSeriesUtils.FEATURES), step=step)

    def normalize(self, normalizer) -> None:
        """
        Normalize the time series
        :param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils._normalize(self.data, normalizer)

    def normalize_inverse(self, normalizer) -> None:
        """
        Inverse normalization of the dataset processed
        :param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils._normalize_inverse(self.data, normalizer)

    def pca(self, model) -> None:
        """
        Apply PCA to the time series
        :param model: PCA model to use
        """
        self.data = TimeSeriesUtils._pca(self.data, model)

    def pca_inverse(self, model) -> None:
        """
        Inverse PCA to the time series
        :param model: PCA model to use
        """
        self.data = TimeSeriesUtils._pca_inverse(self.data, model)
