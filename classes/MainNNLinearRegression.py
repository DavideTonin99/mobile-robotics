import copy

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from torch import nn

from classes.Main import Main
from classes.Params import Params
from models.NNLinearRegression import NNLinearRegression
from models.TimeSeriesUtils import TimeSeriesUtils
from utils.utils import *

from sklearn.preprocessing import StandardScaler


class MainNNLinearRegression(Main):
    """
    Main class implementation for NN Linear Regression
    """

    DEFAULT_PARAMS = {
        'APPLY_MOVING_AVG': False,
        'WINDOW_TYPE': 'sliding',
        'WINDOW_SIZE': 15,
        'WINDOW_STRIDE': 10,
        'APPLY_PCA': True,
        'PCA_COMPONENTS': 7,
        'NORMALIZER_MODEL': StandardScaler(),
        'N_EPOCH': 25000
    }

    def __init__(self, params: Params = None, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        @param params: Dictionary of parameters to be set as attributes
        """
        if params is None:
            params = Params({})
        super().__init__(params)

        for key, value in MainNNLinearRegression.DEFAULT_PARAMS.items():
            if not hasattr(self.params, key) or getattr(self.params, key) is None:
                setattr(self.params, key, value)

        self.ad_params = {}

        self.model = NNLinearRegression(6 * self.params.WINDOW_SIZE, 6 * self.params.WINDOW_SIZE)
        # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.0001)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        self.scheduler = None
        self.device = torch.device("cpu")

    def train(self, t_list: list = None, threshold_params: dict = None, plot_stats: bool = False) -> None:
        x_training = torch.from_numpy(
            self.dataset_train_process.ts_data.data).float().to(self.device)
        y_training = torch.from_numpy(
            self.dataset_train_process.ts_data.data).float().to(self.device)
        x_eval = torch.from_numpy(
            self.dataset_eval_process.ts_data.data).float().to(self.device)
        y_eval = torch.from_numpy(
            self.dataset_eval_process.ts_data.data).float().to(self.device)
        self.model = self.model.to(self.device)

        loss_log = fit_nn_regression(model=self.model, x_training=x_training, y_training=y_training, x_eval=x_eval,
                                     y_eval=y_eval,
                                     optimizer=self.optimizer, scheduler=self.scheduler,
                                     loss_function=self.loss_function, epoch=self.params.N_EPOCH,
                                     print_epoch=True)

        self.ad_params['lower_bound'], self.ad_params['upper_bound'] = compute_quantile_error_threshold(
            errors=loss_log['t'], lower_perc=0.02, upper_perc=0.98)

    def test(self, t_list: list = None, show_plot: bool = True,
             prefix: str = 'test') -> None:
        super().pre_test(t_list=t_list, prefix=prefix)

        # true positive, i.e. truth=anomaly, prediction=anomaly (anomaly detected correctly)
        tp = 0
        # false negative, i.e. truth=anomaly, prediction=nominal (anomaly not detected)
        fn = 0
        # false positive, i.e. truth=nominal, prediction=anomaly (nominal not detected)
        fp = 0
        # true negative, i.e. truth=nominal, prediction=nominal (nominal detected correctly)
        tn = 0

        for ts_name, ts in self.dataset_test.time_series.items():
            ts_process = self.dataset_test_process.time_series[ts_name]
            count_windows_anomaly = 0
            errors_list = []

            anomalies = copy.deepcopy(ts.data)
            for i, row_window in enumerate(ts_process.data):
                row_window = np.array([row_window])
                x = torch.from_numpy(row_window).float().to(self.device)
                y = torch.from_numpy(row_window).float().to(self.device)
                self.model.eval()
                prediction = self.model(x)
                prediction = prediction.cpu().detach().numpy()
                error = (np.square(prediction - row_window).mean(axis=1))
                anomaly_mask = ((error > self.ad_params['upper_bound']) |
                                (error < self.ad_params['lower_bound'])) == True
                if anomaly_mask.any():
                    count_windows_anomaly += 1
                    errors_list.append(error)
                else:
                    anomalies[i * self.params.WINDOW_SIZE:i *
                                                          self.params.WINDOW_SIZE + self.params.WINDOW_SIZE] = np.nan

            curr_traj_predicted_anomaly = count_windows_anomaly > 0  # prediction
            curr_traj_is_anomaly = "anomal" in ts_name  # ground truth
            if curr_traj_is_anomaly and curr_traj_predicted_anomaly:
                tp = tp + 1
            elif curr_traj_is_anomaly and (not curr_traj_predicted_anomaly):
                fn = fn + 1
            elif (not curr_traj_is_anomaly) and curr_traj_predicted_anomaly:
                fp = fp + 1
            elif (not curr_traj_is_anomaly) and (not curr_traj_predicted_anomaly):
                tn = tn + 1

            plot_ts(title=f"NN Linear Regression Figure Test Anomalies {ts_name}", ts_name=f"test_anomalies_{ts_name}",
                    ts={'time_series': ts.data, 'anomalies': anomalies},
                    features=TimeSeriesUtils.PLOT_FEATURES,
                    n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'},
                    markers={'anomalies': 'o'}, save_plot=True, subfolder="nn_linear_regression")
            try:
                plot_scatter(timeseries_1=ts.data, timeseries_2=anomalies,
                             subfolder="nn_linear_regression", filename=ts_name, save_plot=True, show_plot=False,
                             verbose=True)
            except Exception as e:
                pass

            print(
                f"{ts_name}: Anomalies detected: {count_windows_anomaly}, {errors_list}")

        save_stats_txt(tp, fn, fp, tn, subfolder="nn_linear_regression",
                       filename=None, print_stats=True)

    def run(self, train_list: list = None, eval_list: list = None, test_list: list = None,
            show_plot: bool = True) -> None:

        if train_list is not None and len(train_list) > 0:
            super().pre_train(t_list=train_list)
            if eval_list is not None and len(eval_list) > 0:
                super().pre_train(t_list=eval_list, prefix='eval')
            self.train()
        if test_list is not None and len(test_list) > 0:
            self.dataset_test = None
            self.dataset_test_process = None
            self.test(t_list=test_list, show_plot=show_plot, prefix='test')
