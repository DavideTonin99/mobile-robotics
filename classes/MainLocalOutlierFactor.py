import copy

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from classes.Main import Main
from classes.Params import Params
from models.TimeSeriesUtils import TimeSeriesUtils
from utils.utils import *

from sklearn.preprocessing import StandardScaler


class MainLocalOutlierFactor(Main):
    """
    Main class implementation for LocalOutlierFactor
    """

    DEFAULT_PARAMS = {
        'APPLY_MOVING_AVG': False,
        'WINDOW_TYPE': 'sliding',
        'WINDOW_SIZE': 15,
        'WINDOW_STRIDE': 10,
        'APPLY_PCA': True,
        'PCA_COMPONENTS': 7,
        'NORMALIZER_MODEL': StandardScaler(),
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

        for key, value in MainLocalOutlierFactor.DEFAULT_PARAMS.items():
            if not hasattr(self.params, key):
                setattr(self.params, key, value)

        self.pca_model = None
        self.model = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination='auto')
        self.ad_params = {}

    def train(self, t_list: list = None, threshold_params: dict = None, plot_stats: bool = False) -> None:
        if t_list is None:
            t_list = []

        super().pre_train(t_list=t_list)

        # pca model
        if self.params.APPLY_PCA:
            self.pca_model = fit_pca(self.dataset_train_process.ts_data.data, n_components=self.params.PCA_COMPONENTS,
                                     show_plot_variance=False)
            self.dataset_train_process.pca(model=self.pca_model)

        self.model.fit(self.dataset_train_process.ts_data.data)

    def test(self, t_list: list = None, show_plot: bool = True,
             prefix: str = 'test') -> None:
        super().pre_test(t_list=t_list, prefix=prefix)

        if self.params.APPLY_PCA:
            self.dataset_test_process.pca(model=self.pca_model)

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

            predict = self.model.predict(ts.data) == -1

            test_anomalies_mask = np.array([predict] * 6).T
            anomalies = copy.deepcopy(ts.data)

            for i in range(len(predict)):
                if not predict[i]:
                    anomalies[
                        i * self.params.WINDOW_SIZE:i * self.params.WINDOW_STRIDE + self.params.WINDOW_SIZE] = np.nan

            curr_traj_predicted_anomaly = test_anomalies_mask.any()  # is anomaly prediction
            curr_traj_is_anomaly = "anomal" in ts_name  # ground truth
            if curr_traj_is_anomaly and curr_traj_predicted_anomaly:
                tp = tp + 1
            elif curr_traj_is_anomaly and (not curr_traj_predicted_anomaly):
                fn = fn + 1
            elif (not curr_traj_is_anomaly) and curr_traj_predicted_anomaly:
                fp = fp + 1
            elif (not curr_traj_is_anomaly) and (not curr_traj_predicted_anomaly):
                tn = tn + 1

            plot_ts(
                title=f"Local Outlier Factor {'PCA' if self.params.APPLY_PCA else 'NO PCA'} Figure Test Anomalies {ts_name}",
                ts_name=f"{prefix}_anomalies_{ts_name}",
                ts={'time_series': ts.data, 'anomalies': anomalies},
                features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'},
                markers={'anomalies': 'o'}, save_plot=True,
                subfolder=f"local_outlier_{'pca' if self.params.APPLY_PCA else 'no_pca'}")
            try:
                plot_scatter(timeseries_1=ts.data, timeseries_2=anomalies,
                             subfolder=f"local_outlier_{'pca' if self.params.APPLY_PCA else 'no_pca'}", filename=ts_name,
                             save_plot=True,
                             show_plot=False, verbose=True)
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
