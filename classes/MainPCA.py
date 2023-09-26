import copy

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from classes.Main import Main
from classes.Params import Params
from models.TimeSeriesUtils import TimeSeriesUtils
from utils.utils import *

from sklearn.preprocessing import StandardScaler


class MainPCA(Main):
    """
    Main class implementation for PCA
    """

    DEFAULT_PARAMS = {
        'APPLY_MOVING_AVG': False,
        'WINDOW_TYPE': 'sliding',
        'WINDOW_SIZE': 15,
        'WINDOW_STRIDE': 10,
        'PCA_COMPONENTS': 7,
        'NORMALIZER_MODEL': StandardScaler(),
        'THRESHOLD_TYPE': 'quantile',
        'QUANTILE_LOWER_PERCENTAGE': 0.01,
        'QUANTILE_UPPER_PERCENTAGE': 0.99,
    }

    def __init__(self, params: Params = None) -> None:
        """
        Parameters
        ----------
        @param params: Dictionary of parameters to be set as attributes
        """
        if params is None:
            params = Params({})
        super().__init__(params)

        for key, value in MainPCA.DEFAULT_PARAMS.items():
            if not hasattr(self.params, key):
                setattr(self.params, key, value)

        self.pca_model = None
        self.ad_params = {}

    def train(self, t_list: list = None, threshold_params: dict = None, plot_stats: bool = False) -> None:
        if t_list is None:
            t_list = []

        super().pre_train(t_list=t_list)

        # pca model
        self.pca_model = fit_pca(self.dataset_train_process.ts_data.data, n_components=self.params.PCA_COMPONENTS,
                                 show_plot_variance=False)
        self.dataset_train_process.pca(model=self.pca_model)
        self.dataset_train_process.pca_inverse(model=self.pca_model)

        # back to original scale
        self.dataset_train_process.normalize_inverse(normalizer=self.normalizer_model)
        self.dataset_train_process.remove_window(step=self.params.WINDOW_STRIDE)

        errors = np.square(compute_errors(ts_true=self.dataset_train.ts_data.data,
                                          ts_pred=self.dataset_train_process.ts_data.data, calc_abs=False)).mean(axis=1)
        if self.params.THRESHOLD_TYPE == 'quantile':
            self.ad_params['lower_bound'], self.ad_params['upper_bound'] = compute_quantile_error_threshold(
                errors=errors, lower_perc=self.params.QUANTILE_LOWER_PERCENTAGE,
                upper_perc=self.params.QUANTILE_UPPER_PERCENTAGE)
        elif self.params.THRESHOLD_TYPE == 'svm':
            self.ad_params['one_class_model'] = OneClassSVM(gamma="auto").fit(
                errors.reshape(-1, 1))
        # one_class_model = OneClassSVM(gamma="scale", nu=0.01, kernel='poly').fit(train_errors.reshape(-1, 1))
        elif self.params.THRESHOLD_TYPE == 'local_outlier_factor':
            self.ad_params['local_outlier_factor_model'] = LocalOutlierFactor(
                n_neighbors=100, novelty=True, contamination='auto').fit(errors.reshape(-1, 1))

        plot_ts(title="PCA Figure Train", ts_name="train",
                ts={'start': self.dataset_train.ts_data.data, 'end': self.dataset_train_process.ts_data.data},
                features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True,
                subfolder=f"pca_{self.params.THRESHOLD_TYPE}")

    def test(self, t_list: list = None, corruption_params: dict = None, show_plot: bool = True,
             prefix: str = 'test') -> None:
        super().pre_test(t_list=t_list, prefix=prefix)

        self.dataset_test_process.pca(model=self.pca_model)

        self.dataset_test_process.pca_inverse(model=self.pca_model)

        self.dataset_test_process.normalize_inverse(normalizer=self.normalizer_model)
        self.dataset_test_process.remove_window(step=self.params.WINDOW_STRIDE)

        # true positive, i.e. truth=anomaly, prediction=anomaly (anomaly detected correctly)
        tp = 0
        # false negative, i.e. truth=anomaly, prediction=nominal (anomaly not detected)
        fn = 0
        # false positive, i.e. truth=nominal, prediction=anomaly (nominal not detected)
        fp = 0
        # true negative, i.e. truth=nominal, prediction=nominal (nominal detected correctly)
        tn = 0

        for ts_name, ts in self.dataset_test.time_series.items():
            ts_pred = self.dataset_test_process.time_series[ts_name]

            plot_ts(title=f"PCA Figure {prefix} {ts_name}", ts_name=f"{prefix}_{ts_name}",
                    ts={'start': ts.data, 'end': ts_pred.data}, features=TimeSeriesUtils.PLOT_FEATURES,
                    n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True,
                    subfolder=f"pca_{self.params.THRESHOLD_TYPE}")

            errors = compute_errors(ts_true=ts.data, ts_pred=ts_pred.data, calc_abs=False)
            errors = np.square(errors).mean(axis=1)

            anomaly_mask = None
            if self.params.THRESHOLD_TYPE == 'quantile':
                anomaly_mask = np.logical_or(errors < self.ad_params['lower_bound'],
                                             errors > self.ad_params['upper_bound'])
            elif self.params.THRESHOLD_TYPE == 'svm':
                anomaly_mask = self.ad_params['one_class_model'].predict(
                    errors.reshape(-1, 1)) == -1
            elif self.params.THRESHOLD_TYPE == 'local_outlier_factor':
                anomaly_mask = self.ad_params['local_outlier_factor_model'].predict(
                    errors.reshape(-1, 1)) == -1

            if anomaly_mask is not None:
                print(prefix, np.count_nonzero(anomaly_mask))

                anomalies = copy.deepcopy(ts.data)
                anomalies[np.logical_not(anomaly_mask)] = np.nan

                curr_traj_predicted_anomaly = np.count_nonzero(
                    anomaly_mask) > 0  # prediction
                curr_traj_is_anomaly = "anomal" in ts_name  # ground truth
                if curr_traj_is_anomaly and curr_traj_predicted_anomaly:
                    tp = tp + 1
                elif curr_traj_is_anomaly and (not curr_traj_predicted_anomaly):
                    fn = fn + 1
                elif (not curr_traj_is_anomaly) and curr_traj_predicted_anomaly:
                    fp = fp + 1
                elif (not curr_traj_is_anomaly) and (not curr_traj_predicted_anomaly):
                    tn = tn + 1

                plot_ts(title=f"PCA Figure {prefix} Anomalies {ts_name}", ts_name=f"{prefix}_anomalies_{ts_name}",
                        ts={'time_series': ts.data, 'anomalies': anomalies},
                        features=TimeSeriesUtils.PLOT_FEATURES,
                        n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'},
                        markers={'anomalies': 'o'}, save_plot=True, subfolder=f"pca_{self.params.THRESHOLD_TYPE}")
                try:
                    plot_scatter(timeseries_1=ts.data, timeseries_2=anomalies,
                                 subfolder=f"pca_{self.params.THRESHOLD_TYPE}", filename=ts_name, save_plot=True,
                                 show_plot=False,
                                 verbose=True)
                except Exception as e:
                    pass
        save_stats_txt(
            tp, fn, fp, tn, subfolder=f"pca_{prefix}_{self.params.THRESHOLD_TYPE}", filename=None, print_stats=True)

    def run(self, train_list: list = None, eval_list: list = None, test_list: list = None,
            show_plot: bool = True) -> None:

        if train_list is not None and len(train_list) > 0:
            self.train(t_list=train_list)
        if eval_list is not None and len(eval_list) > 0:
            self.dataset_test = None
            self.dataset_test_process = None
            self.test(t_list=eval_list, show_plot=show_plot, prefix='eval')
        if test_list is not None and len(test_list) > 0:
            self.dataset_test = None
            self.dataset_test_process = None
            self.test(t_list=test_list, show_plot=show_plot, prefix='test')
