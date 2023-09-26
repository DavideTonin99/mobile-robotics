from utils import *


class TimeSeriesUtils:
    FEATURES = ['x', 'y', 'theta', 'LS', 'LC', 'LD']
    PLOT_FEATURES = {'pose': ['x', 'y', 'theta'], 'lidar': ['LS', 'LC', 'LD']}

    def __init__(self) -> None:
        """
        **************************************************
        * Class TimeSeriesUtils                          *
        **************************************************
        """
        pass

    @staticmethod
    def ts_moving_average(data, step) -> np.array:
        """
        Apply moving average to the time series
        :param data: data of the time series
        :param step: step of the moving average
        """
        result = None

        for i in range(data.shape[1]):
            if result is None:
                result = np.array([np.convolve(data[:, i], np.ones(step), 'valid') / step])
            else:
                result = np.concatenate([result, [np.convolve(data[:, i], np.ones(step), 'valid') / step]])
        return result.T

    @staticmethod
    def ts_normalize(data, normalizer) -> np.array:
        """
        Normalize the time series
        :param data: time series of the time series
        :param normalizer: normalizer to use
        """
        return normalizer.transform(data)

    @staticmethod
    def ts_normalize_inverse(data, normalizer) -> np.array:
        """
        Inverse normalization the time series
        :param data: time series of the time series
        :param normalizer: normalizer to use
        """
        return normalizer.inverse_transform(data)

    @staticmethod
    def ts_sliding_window(data, window_size, stride=1) -> np.array:
        """
        Add a sliding window to the time series
        :param data: data of the time series
        :param window_size: size of the window
        :param stride: stride of the window
        """
        indexer = np.arange(window_size * len(TimeSeriesUtils.FEATURES))[
                  None, :] + stride * len(TimeSeriesUtils.FEATURES) * np.arange(
            (len(data) - (window_size - stride)) // (stride))[:, None]

        return data.flatten()[indexer]

    @staticmethod
    def ts_tumbling_window(data, window_size=None) -> np.array:
        """
        Add a tumbling window to the time series
        :param data: data of the time series
        :param window_size: size of the window
        """
        return TimeSeriesUtils.ts_sliding_window(data, window_size, window_size)

    @staticmethod
    def ts_remove_window(data: np.array, n_features: int, step: int = 1, overlap_keep: str = 'end') -> np.array:
        """
        Remove a sliding window from the time series
        :param data: data of the time series
        :param n_features: number of features
        :param step: step of the window
        :param overlap_keep: if 'end', keep the last part of the window, if 'start', keep the first part of the window
        """
        if overlap_keep == 'end':
            result = np.concatenate([data[0].flatten(), data[1:, -(n_features * step):].flatten()]).reshape(-1,
                                                                                                            n_features)
        else:
            result = np.concatenate([data[:-1, :(n_features * step)].flatten(), data[-1].flatten()]).reshape(-1,
                                                                                                             n_features)
        return result

    @staticmethod
    def ts_pca(data, model) -> np.array:
        """
        Apply PCA to the time series
        :param data: data of the time series
        :param model: PCA model to use    
        """
        return model.transform(data)

    @staticmethod
    def ts_pca_inverse(data, model) -> np.array:
        """
        Inverse PCA to the time series
        :param data: data of the time series
        :param model: PCA model to use    
        """
        return model.inverse_transform(data)

    @staticmethod
    def ts_load_from_features_csv(
            name,
            remove_header=False,
            data_base_path=None,
    ) -> np.array:
        """
        Load the time series from the csv files
        :param name: name of the time series
        :param output_path: path where to save the time series
        :param data_base_path: path where to find the csv files
        """
        result = None

        data_base_path = data_base_path or DATA_BASE_PATH

        if os.path.isdir(os.path.join(data_base_path, name)):
            result = pd.read_csv(os.path.join(
                data_base_path, name, f'ml_data_{name}.csv'), sep=",", header=None,
                names=TimeSeriesUtils.FEATURES).to_numpy()
            if remove_header:
                result = result[1:]
        else:
            raise Exception(f"Time series {name} not found")
        return result
