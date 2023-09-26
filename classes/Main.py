from abc import abstractmethod, ABC

import numpy as np
from models.TimeSeries import TimeSeries
from models.Dataset import Dataset
import copy
from classes.Params import Params


class Main(ABC):
    """
    Abstract Main class
    """

    def __init__(self, params: Params) -> None:
        """
        Parameters
        ----------
        @params : Params : Parameters of the model
        """
        self.params = params
        self.normalizer_model = params.NORMALIZER_MODEL
        self.ad_model = None
        self.dataset_train = None
        self.dataset_train_process = None
        self.dataset_eval = None
        self.dataset_eval_process = None
        self.dataset_test = None
        self.dataset_test_process = None
        self.errors = {}

    def get_dataset_name(self, prefix: str):
        """
        Get the name of the dataset
        @param prefix:
        @return:
        """
        name = prefix
        if self.params.WINDOW_TYPE is not None:
            name += f' {self.params.WINDOW_TYPE} window'
        if self.params.WINDOW_SIZE is not None:
            name += f', window size = {self.params.WINDOW_SIZE}'
        if self.params.WINDOW_STRIDE is not None:
            name += f', window stride = {self.params.WINDOW_STRIDE}'
        return name

    @staticmethod
    def load_data(t_list: list, params: Params) -> dict:
        """
        Load the data
        @param t_list: list of time series
        @param params: parameters of the model
        @return: dictionary of time series
        """
        data = {}
        for t_name in t_list:
            t = TimeSeries(name=t_name, auto_load=True)
            data[t_name] = t
        return data

    def prepare_dataset(self, name: str, data: dict, is_train: bool) -> [Dataset,
                                                                         Dataset]:
        """
        Prepare the dataset
        @param name: name of the dataset
        @param data: data of the dataset
        @param is_train: if True, the dataset is a training dataset
        @return: [dataset, dataset_process]
        """
        window_params = {
            'window_type': self.params.WINDOW_TYPE,
            'window_size': self.params.WINDOW_SIZE,
            'window_stride': self.params.WINDOW_STRIDE
        }
        dataset = Dataset(name, time_series=data, is_train=is_train)
        if self.params.APPLY_MOVING_AVG:
            dataset.moving_average(step=self.params.MOVING_AVG_STEP)
        dataset.prepare_for_window(**window_params)

        dataset_process = copy.deepcopy(dataset)

        dataset_process.add_window(**window_params)
        if is_train:
            self.normalizer_model.fit(dataset_process.ts_data.data)
        dataset_process.normalize(normalizer=self.normalizer_model)

        return [dataset, dataset_process]

    def pre_train(self, t_list: list = None, prefix: str = 'train') -> None:
        """
        Train the model
        @param t_list: list of training time series
        """
        if t_list is None:
            t_list = []
        data = self.load_data(t_list=t_list, params=self.params)
        name = self.get_dataset_name(prefix=prefix)

        if prefix == 'train':
            self.dataset_train, self.dataset_train_process = self.prepare_dataset(name=name, data=data, is_train=True)
        else:
            self.dataset_eval, self.dataset_eval_process = self.prepare_dataset(name=name, data=data, is_train=True)

    def pre_test(self, t_list: list = None, prefix: str = 'test') -> None:
        """
        Operations pre-test: load data, create dataset, normalize, apply sliding window
        @param t_list: list of test time series
        @param prefix: prefix of the dataset name
        """
        if t_list is None:
            t_list = []
        data = self.load_data(t_list=t_list, params=self.params)
        name = self.get_dataset_name(prefix=prefix)

        if prefix == 'test':
            self.dataset_test, self.dataset_test_process = self.prepare_dataset(name=name, data=data, is_train=False)
        else:
            self.dataset_eval, self.dataset_eval = self.prepare_dataset(name=name, data=data, is_train=False)

    @abstractmethod
    def train(self, t_list: list) -> None:
        """
        Train the model
        @param t_list: list of training time series
        @raises NotImplementedError
        """
        pass

    @abstractmethod
    def test(self, t_list: list, show_plot: bool = True, prefix: str = 'test') -> None:
        """
        Test the model
        @param t_list: list of test time series
        @param show_plot: if True, show the plot
        @param prefix: prefix of the dataset name
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Run execution
        """
        pass
