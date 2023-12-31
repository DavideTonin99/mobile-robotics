import numpy as np
from models.TimeSeries import TimeSeries


class Dataset:

    def __init__(self, name: str, time_series: dict, is_train: bool) -> None:
        """
        **************************************************
        * Class Dataset                                  *
        **************************************************
        :param name: name of the dataset (Example: train, test, nominal, anomaly, ...)
        :param time series: dictionary of time series of the dataset -> [{name: TimeSeries}, ...]
        :return: None
        """
        self.name = name
        self.window_size = None
        self.is_train = is_train

        self.time_series = time_series
        self.ts_data = TimeSeries(name=name, data=np.concatenate(
            [ts.data for ts in self.time_series.values()], axis=0))

    def get_ts(self, name: str) -> TimeSeries:
        """
        Get a time series from the dataset
        :param name: name of the time series
        :return: time series
        """
        return self.time_series[name]

    def moving_average(self, step: int, ts_name: str = None) -> None:
        """
        Apply moving average to the dataset
        :param step: step of the moving average
        :param ts_name: name of the time series to apply moving average, if is None apply moving average to all the dataset
        :return: None
        """
        if not self.is_train:
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].moving_average(step)
                else:
                    raise Exception("Time Series not found")
            else:
                for key, value in self.time_series.items():
                    value.moving_average(step)
                    self.time_series[key] = value
            self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
                [ts.data for ts in self.time_series.values()], axis=0))
        else:
            self.ts_data.moving_average(step)
            sum_len = 0
            for key, value in self.time_series.items():
                len_ts = len(value.data)
                self.time_series[key].data = self.ts_data.data[sum_len:sum_len + len_ts]
                sum_len += len_ts

    def prepare_for_window(self, window_type: str, window_size: int, window_stride: int, ts_name: str = None) -> None:
        """
        Add a window to the trajectories of the dataset
        @param window_type: type of window to add ("sliding" | "tumbling")
        @param window_size: size of the window
        @param window_stride: stride of the window
        @param ts_name: name of the time series to add the window, if is None add the window to all the dataset
        @return: None
        @raise Exception: time series not found
        """
        if ts_name is not None:
            if ts_name in self.time_series.keys():
                self.time_series[ts_name].prepare_for_window(window_type=window_type, window_size=window_size,
                                                             window_stride=window_stride)
            else:
                raise Exception("Time Series not found")
        else:
            for ts_name in self.time_series.keys():
                self.time_series[ts_name].prepare_for_window(window_type=window_type, window_size=window_size,
                                                             window_stride=window_stride)

        if self.is_train:
            self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
                [ts.data for ts in self.time_series.values()], axis=0))

    def add_window(self, ts_name: str = None, window_type: str = "sliding", window_size: int = None,
                   window_stride: int = 1) -> None:
        """
        Add a window to the trajectories of the dataset
        :param ts_name: name of the time series to add the window, if is None add the window to all the dataset
        :param window_type: type of window to add ("sliding" | "tumbling")
        :param window_size: size of the window
        :param window_stride: stride of the window
        :return: None
        Exception: window type not supported
        """
        if window_size is not None:
            self.window_size = window_size

            if window_type != "sliding" and window_type != "tumbling":
                raise Exception("Window type not supported")

            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name] = self.time_series[ts_name].add_window(
                        window_type, window_size, window_stride)
                else:
                    raise Exception("Time Series not found")
            else:
                for key, value in self.time_series.items():
                    value.add_window(window_type, window_size, window_stride)
                    self.time_series[key] = value
            self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
                [ts.data for ts in self.time_series.values()], axis=0))
        else:
            raise Exception("Window size not specified")

    def remove_window(self, ts_name: str = None, step: int = 1) -> None:
        """
        Remove a window from the dataset (sliding or tumbling)
        :param ts_name: name of the time series to remove the window, if is None remove the window to all the dataset
        :param step: step of the window
        :return: None
        """
        if ts_name is not None:
            if ts_name in self.time_series.keys():
                self.time_series[ts_name] = self.time_series[ts_name].remove_window(step)
            else:
                raise Exception("Time Series not found")
        else:
            for key, value in self.time_series.items():
                value.remove_window(step)
                self.time_series[key] = value
        self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
            [ts.data for ts in self.time_series.values()], axis=0))

    def normalize(self, normalizer, ts_name: str = None) -> None:
        """
        Normalize the dataset
        :param normalizer: normalizer to use
        :param ts_name: name of the time series to normalize, if is None normalize all the dataset
        :return: None
        """
        if not self.is_train:
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name] = self.time_series[ts_name].normalize(normalizer)
                else:
                    raise Exception("Time Series not found")
            else:
                for key, value in self.time_series.items():
                    value.normalize(normalizer)
                    self.time_series[key] = value
        else:
            self.ts_data.normalize(normalizer)
            sum_len = 0
            for key, value in self.time_series.items():
                len_ts = len(value.data)
                self.time_series[key].data = self.ts_data.data[sum_len:sum_len + len_ts]
                sum_len += len_ts

    def normalize_inverse(self, normalizer, ts_name: str = None) -> None:
        """
        Inverse normalization of the dataset processed
        :param normalizer: normalizer to use
        :param ts_name: name of the time series to normalize, if is None normalize all the dataset
        :return: None
        """
        if not self.is_train:
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name] = self.time_series[ts_name].normalize_inverse(
                        normalizer)
                else:
                    raise Exception("Time Series not found")
            else:
                for key, value in self.time_series.items():
                    value.normalize_inverse(normalizer)
                    self.time_series[key] = value
        else:
            self.ts_data.normalize_inverse(normalizer)
            sum_len = 0
            for key, value in self.time_series.items():
                len_ts = len(value.data)
                self.time_series[key].data = self.ts_data.data[sum_len:sum_len + len_ts]
                sum_len += len_ts

    def pca(self, model, ts_name: str = None) -> None:
        """
        Apply PCA to the dataset
        :param model: PCA model to use
        :param ts_name: name of the time series to apply PCA, if is None apply PCA to all the dataset
        :return: None
        """
        if not self.is_train:
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name] = self.time_series[ts_name].pca(model)
                else:
                    raise Exception("Time Series not found")
            else:
                for key, value in self.time_series.items():
                    value.pca(model)
                    self.time_series[key] = value
        else:
            self.ts_data.pca(model)
            sum_len = 0
            for key, value in self.time_series.items():
                len_ts = len(value.data)
                self.time_series[key].data = self.ts_data.data[sum_len:sum_len + len_ts]
                sum_len += len_ts

    def pca_inverse(self, model, ts_name: str = None) -> None:
        """
        Inverse PCA to the dataset
        :param model: PCA model to use
        :param ts_name: name of the time series to inverse PCA, if is None inverse PCA to all the dataset
        :return: None
        """
        if not self.is_train:
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name] = self.time_series[ts_name].pca_inverse(model)
                else:
                    raise Exception("Time Series not found")
            else:
                for key, value in self.time_series.items():
                    value.pca_inverse(model)
                    self.time_series[key] = value
        else:
            self.ts_data.pca_inverse(model)
            sum_len = 0
            for key, value in self.time_series.items():
                len_ts = len(value.data)
                self.time_series[key].data = self.ts_data.data[sum_len:sum_len + len_ts]
                sum_len += len_ts
