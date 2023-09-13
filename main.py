from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from TimeSeriesUtils import *
from TimeSeries import *
from utils import *
import copy
from NNLinearRegression import *


def main_pca(dataset_train, scaler_model, window_type, window_size, window_stride, pca_components, show_plot_variance=True, threshold_method='quantile', dataset_eval=None, dataset_test=None):
    """
    Main function for the PCA
    :param dataset_train: training dataset
    :param scaler_model: scaler model
    :param window_type: window type
    :param window_size: window size
    :param window_stride: window stride
    :param pca_components: number of components to keep
    :param show_plot_variance: show the plot of the variance
    :param dataset_eval: evaluation dataset
    :param dataset_test: test dataset
    """
    dataset_train.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_train.remove_window(step=window_stride)

    dataset_eval.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_eval.remove_window(step=window_stride)

    dataset_test.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_test.remove_window(step=window_stride)

    dataset_train_process = copy.deepcopy(dataset_train)
    if dataset_eval is not None:
        dataset_eval_process = copy.deepcopy(dataset_eval)
    if dataset_test is not None:
        dataset_test_process = copy.deepcopy(dataset_test)

    # TRAININ SECTION
    dataset_train_process.add_window(
        type=window_type, window_size=window_size, stride=window_stride)

    scaler_model.fit(dataset_train_process.ts_data.data)
    dataset_train_process.normalize(normalizer=scaler_model, mode='complete')

    model = fit_pca(dataset_train_process.ts_data.data, n_components=pca_components,
                    show_plot_variance=show_plot_variance)

    dataset_train_process.pca(model=model, mode='complete')
    dataset_train_process.pca_inverse(model=model, mode='complete')
    dataset_train_process.normalize_inverse(
        normalizer=scaler_model, mode='complete')
    dataset_train_process.remove_window(step=window_stride)

    train_errors = (np.square(dataset_train_process.ts_data.data - dataset_train.ts_data.data).mean(axis=1))
    if threshold_method == 'quantile':
        pc_lower_bound, pc_upper_bound = compute_quantile_error_threshold(errors=train_errors, lower_perc=0.05, upper_perc=0.95)
    elif threshold_method == 'svm':
        one_class_model = OneClassSVM(gamma="auto").fit(train_errors.reshape(-1, 1))
        # one_class_model = OneClassSVM(gamma="scale", nu=0.01, kernel='poly').fit(train_errors.reshape(-1, 1))
    elif threshold_method == 'local_outlier_factor':
        local_outlier_factor_model = LocalOutlierFactor(n_neighbors=100, novelty=True, contamination='auto').fit(train_errors.reshape(-1, 1))
    
    plot_ts(title="PCA Figure Train", ts_name="train", ts={'start': dataset_train.ts_data.data, 'end': dataset_train_process.ts_data.data}, features=TimeSeriesUtils.PLOT_FEATURES,
            n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True, subfolder=f"pca_{threshold_method}")

    if dataset_eval is not None:
        dataset_eval_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_eval_process.normalize(normalizer=scaler_model, mode='single')

        # apply pca on eval dataset
        dataset_eval_process.pca(model=model, mode='single')
        dataset_eval_process.pca_inverse(model=model, mode='single')

        # back to original scale
        dataset_eval_process.normalize_inverse(
            normalizer=scaler_model, mode='single')
        dataset_eval_process.remove_window(step=window_stride)

        for ts_name, ts in dataset_eval_process.time_series.items():
            plot_ts(title=f"PCA Figure Eval {ts_name}", ts_name=f"eval_{ts_name}", ts={'start': dataset_eval.time_series[ts_name].data, 'end': dataset_eval_process.time_series[ts_name].data}, features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True, subfolder=f"pca_{threshold_method}")

            eval_errors = (np.square(dataset_eval_process.time_series[ts_name].data - dataset_eval.time_series[ts_name].data).mean(axis=1))
            
            eval_anomalies_mask = None
            if threshold_method == 'quantile':
                eval_anomalies_mask = ((eval_errors > pc_upper_bound) | (eval_errors < pc_lower_bound)) == True
            elif threshold_method == 'svm':
                eval_anomalies_mask = one_class_model.predict(eval_errors.reshape(-1, 1)) == -1
            elif threshold_method == 'local_outlier_factor':
                eval_anomalies_mask = local_outlier_factor_model.predict(eval_errors.reshape(-1, 1)) == -1

            if eval_anomalies_mask is not None:            
                print("eval", np.count_nonzero(eval_anomalies_mask))
                    
                eval_anomalies_mask = np.array([eval_anomalies_mask]*6).T
                anomalies = copy.deepcopy(dataset_eval.time_series[ts_name].data)
                anomalies[~eval_anomalies_mask] = np.nan
            
                plot_ts(title=f"PCA Figure Eval Anomalies {ts_name}", ts_name=f"eval_anomalies_{ts_name}", ts={'time_series': dataset_eval.time_series[ts_name].data, 'anomalies': anomalies}, features=TimeSeriesUtils.PLOT_FEATURES,
                        n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'}, markers={'anomalies': 'o'}, save_plot=True, subfolder=f"pca_{threshold_method}")

    # TEST SECTION
    if dataset_test is not None:
        dataset_test_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_test_process.normalize(normalizer=scaler_model, mode='single')

        # apply pca on test dataset
        dataset_test_process.pca(model=model, mode='single')
        dataset_test_process.pca_inverse(model=model, mode='single')

        # back to original scale
        dataset_test_process.normalize_inverse(
            normalizer=scaler_model, mode='single')
        dataset_test_process.remove_window(step=window_stride)

        tp = 0 # true positive, i.e. truth=anomaly, prediction=anomaly (anomaly detected correctly)
        fn = 0 # false negative, i.e. truth=anomaly, prediction=nominal (anomaly not detected)
        fp = 0 # false positive, i.e. truth=nominal, prediction=anomaly (nominal not detected)
        tn = 0 # true negative, i.e. truth=nominal, prediction=nominal (nominal detected correctly)
        for ts_name, ts in dataset_test_process.time_series.items():
            plot_ts(title=f"PCA Figure Test {ts_name}", ts_name=f"test_{ts_name}", ts={'start': dataset_test.time_series[ts_name].data, 'end': dataset_test_process.time_series[ts_name].data}, features=TimeSeriesUtils.PLOT_FEATURES,
                    n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True, subfolder=f"pca_{threshold_method}")
            
            test_errors = (np.square(dataset_test_process.time_series[ts_name].data - dataset_test.time_series[ts_name].data).mean(axis=1))

            test_anomalies_mask = None
            if threshold_method == 'quantile':
                test_anomalies_mask = ((test_errors > pc_upper_bound) | (test_errors < pc_lower_bound)) == True
            elif threshold_method == 'svm':
                test_anomalies_mask = one_class_model.predict(test_errors.reshape(-1, 1)) == -1
            elif threshold_method == 'local_outlier_factor':
                test_anomalies_mask = local_outlier_factor_model.predict(test_errors.reshape(-1, 1)) == -1
            
            if test_anomalies_mask is not None:
                print("test", np.count_nonzero(test_anomalies_mask))

                curr_traj_predicted_anomaly = np.count_nonzero(test_anomalies_mask) > 0 # prediction
                curr_traj_is_anomaly = "anomal" in ts_name # ground truth
                if curr_traj_is_anomaly and curr_traj_predicted_anomaly:
                    tp = tp + 1
                elif curr_traj_is_anomaly and (not curr_traj_predicted_anomaly):
                    fn = fn + 1
                elif (not curr_traj_is_anomaly) and curr_traj_predicted_anomaly:
                    fp = fp + 1
                elif (not curr_traj_is_anomaly) and (not curr_traj_predicted_anomaly):
                    tn = tn + 1

                test_anomalies_mask = np.array([test_anomalies_mask]*6).T
                anomalies = copy.deepcopy( dataset_test.time_series[ts_name].data)
                anomalies[~test_anomalies_mask] = np.nan

                plot_ts(title=f"PCA Figure Test Anomalies {ts_name}", ts_name=f"test_anomalies_{ts_name}", ts={'time_series': dataset_test.time_series[ts_name].data, 'anomalies': anomalies}, features=TimeSeriesUtils.PLOT_FEATURES,
                    n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'}, markers={'anomalies': 'o'}, save_plot=True, subfolder=f"pca_{threshold_method}")

        save_stats_txt(tp, fn, fp, tn, subfolder=f"pca_{threshold_method}", filename=None, print_stats=True)

def main_nn_linear_regression(dataset_train, model, scaler_model, window_type, window_size, window_stride, device, optimizer, loss_function, epoch, scheduler=None, dataset_eval=None, dataset_test=None):
    """
    Main function for the NN Linear Regression
    :param dataset_train: training dataset
    :param model: model
    :param scaler_model: scaler model
    :param window_type: window type
    :param window_size: window size
    :param window_stride: window stride
    :param device: device
    :param optimizer: optimizer
    :param loss_function: loss function
    :param epoch: number of epochs
    :param scheduler: scheduler
    :param dataset_eval: evaluation dataset
    :param dataset_test: test dataset
    """
    dataset_train_process = copy.deepcopy(dataset_train)
    if dataset_eval is not None:
        dataset_eval_process = copy.deepcopy(dataset_eval)
    if dataset_test is not None:
        dataset_test_process = copy.deepcopy(dataset_test)

    dataset_train_process.add_window(
        type=window_type, window_size=window_size, stride=window_stride)

    scaler_model.fit(dataset_train_process.ts_data.data)
    dataset_train_process.normalize(normalizer=scaler_model, mode='complete')

    if dataset_eval is not None:
        dataset_eval_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_eval_process.normalize(normalizer=scaler_model, mode='complete')

    x_training = torch.from_numpy(
        dataset_train_process.ts_data.data).float().to(device)
    y_training = torch.from_numpy(
        dataset_train_process.ts_data.data).float().to(device)
    x_eval = torch.from_numpy(
        dataset_eval_process.ts_data.data).float().to(device)
    y_eval = torch.from_numpy(
        dataset_eval_process.ts_data.data).float().to(device)
    model = model.to(device)

    loss_log = fit(model=model, x_training=x_training, y_training=y_training, x_eval=x_eval, y_eval=y_eval,
                   optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, epoch=epoch, print_epoch=True)

    pc_lower_bound, pc_upper_bound = compute_quantile_error_threshold(errors=loss_log['t'], lower_perc=0.02, upper_perc=0.98)
    # print(f"* Lower Bound: {pc_lower_bound}")
    # print(f"* Upper Bound: {pc_upper_bound}")

    if dataset_test is not None:
        dataset_test_process.add_window(
            type='tumbling', window_size=window_size)
        dataset_test_process.normalize(normalizer=scaler_model, mode='single')
        tp = 0  # true positive, i.e. truth=anomaly, prediction=anomaly (anomaly detected correctly)
        fn = 0  # false negative, i.e. truth=anomaly, prediction=nominal (anomaly not detected)
        fp = 0  # false positive, i.e. truth=nominal, prediction=anomaly (nominal not detected)
        tn = 0  # true negative, i.e. truth=nominal, prediction=nominal (nominal detected correctly)

        for ts_name, t in dataset_test_process.time_series.items():
            count_windows_anomaly = 0
            errors_list = []

            anomalies = copy.deepcopy(dataset_test.time_series[ts_name].data)
            for i, row_window in enumerate(t.data):
                row_window = np.array([row_window])
                x = torch.from_numpy(row_window).float().to(device)
                y = torch.from_numpy(row_window).float().to(device)
                model.eval()
                prediction = model(x)
                prediction = prediction.cpu().detach().numpy()
                error = (np.square(prediction - row_window).mean(axis=1))
                anomalies_mask = ((error > pc_upper_bound) | (error < pc_lower_bound)) == True
                if anomalies_mask.any():
                    count_windows_anomaly += 1
                    errors_list.append(error)
                else:
                    anomalies[i*window_size:i*window_size + window_size] = np.nan

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

            plot_ts(title=f"NN Linear Regression Figure Test Anomalies {ts_name}", ts_name=f"test_anomalies_{ts_name}", ts={'time_series': dataset_test.time_series[ts_name].data, 'anomalies': anomalies}, features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'}, markers={'anomalies': 'o'}, save_plot=True, subfolder="nn_linear_regression")

            print(f"{ts_name}: Anomalies detected: {count_windows_anomaly}, {errors_list}")

        save_stats_txt(tp, fn, fp, tn, subfolder="nn_linear_regression", filename=None, print_stats=True)


def main_svm(dataset_train, model, scaler_model, window_type, window_size, window_stride, with_pca, pca_components, show_plot_variance=True, dataset_eval=None, dataset_test=None):
    """
    Main function for the SVM
    :param dataset_train: training dataset
    :param model: model
    :param scaler_model: scaler model
    :param window_type: window type
    :param window_size: window size
    :param window_stride: window stride
    :param with_pca: apply pca
    :param pca_components: number of components to keep
    :param show_plot_variance: show the plot of the variance
    :param dataset_eval: evaluation dataset
    :param dataset_test: test dataset
    """
    dataset_train.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_train.remove_window(step=window_stride)

    dataset_eval.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_eval.remove_window(step=window_stride)

    dataset_test.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_test.remove_window(step=window_stride)

    dataset_train_process = copy.deepcopy(dataset_train)
    if dataset_eval is not None:
        dataset_eval_process = copy.deepcopy(dataset_eval)
    if dataset_test is not None:
        dataset_test_process = copy.deepcopy(dataset_test)

    # TRAININ SECTION
    dataset_train_process.add_window(
        type=window_type, window_size=window_size, stride=window_stride)

    scaler_model.fit(dataset_train_process.ts_data.data)
    dataset_train_process.normalize(normalizer=scaler_model, mode='complete')

    pca_model = None
    if with_pca:
        pca_model = fit_pca(dataset_train_process.ts_data.data, n_components=pca_components,
                    show_plot_variance=show_plot_variance)
        dataset_train_process.pca(model=pca_model, mode='complete')

    model = model.fit(dataset_train_process.ts_data.data)

    if dataset_eval is not None:
        dataset_eval_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_eval_process.normalize(normalizer=scaler_model, mode='single')

        # apply pca on eval dataset
        if with_pca:
            dataset_eval_process.pca(model=pca_model, mode='single')

        for ts_name, ts in dataset_eval_process.time_series.items():
            one_class_predict = model.predict(ts.data) == -1

            eval_anomalies_mask = np.array([one_class_predict]*6).T
            anomalies = copy.deepcopy(dataset_eval.time_series[ts_name].data)

            for i in range(len(one_class_predict)):
                if not one_class_predict[i]:
                    anomalies[i*window_size:i*window_size + window_size] = np.nan

            plot_ts(title=f"OneClass SVM {'PCA' if with_pca else 'NO PCA'} Figure Eval Anomalies {ts_name}", ts_name=f"eval_anomalies_{ts_name}", ts={'time_series': dataset_eval.time_series[ts_name].data, 'anomalies': anomalies}, features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'}, markers={'anomalies': 'o'}, save_plot=True, subfolder=f"svm_{'pca' if with_pca else 'no_pca'}")

    # TEST SECTION
    if dataset_test is not None:
        dataset_test_process.add_window(
            type='tumbling', window_size=window_size)
        dataset_test_process.normalize(normalizer=scaler_model, mode='single')

        if with_pca:
            # apply pca on train dataset
            dataset_test_process.pca(model=pca_model, mode='single')

        tp = 0  # true positive, i.e. truth=anomaly, prediction=anomaly (anomaly detected correctly)
        fn = 0  # false negative, i.e. truth=anomaly, prediction=nominal (anomaly not detected)
        fp = 0  # false positive, i.e. truth=nominal, prediction=anomaly (nominal not detected)
        tn = 0  # true negative, i.e. truth=nominal, prediction=nominal (nominal detected correctly)

        for ts_name, ts in dataset_test_process.time_series.items():
            one_class_predict = model.predict(ts.data) == -1

            test_anomalies_mask = np.array([one_class_predict]*6).T
            anomalies = copy.deepcopy(dataset_test.time_series[ts_name].data)

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

            for i in range(len(one_class_predict)):
                if not one_class_predict[i]:
                    anomalies[i*window_size:i*window_size + window_size] = np.nan

            plot_ts(title=f"OneClass SVM {'PCA' if with_pca else 'NO PCA'} Figure Test Anomalies {ts_name}", ts_name=f"test_anomalies_{ts_name}", ts={'time_series': dataset_test.time_series[ts_name].data, 'anomalies': anomalies}, features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'}, markers={'anomalies': 'o'}, save_plot=True, subfolder=f"svm_{'pca' if with_pca else 'no_pca'}")

        save_stats_txt(tp, fn, fp, tn, subfolder=f"svm_{'pca' if with_pca else 'no_pca'}", filename=None, print_stats=True)

def main_local_outlier_factor(dataset_train, model, scaler_model, window_type, window_size, window_stride, with_pca, pca_components, show_plot_variance=True, dataset_eval=None, dataset_test=None):
    """
    Main function for the Local Outlier Factor
    :param dataset_train: training dataset
    :param model: model
    :param scaler_model: scaler model
    :param window_type: window type
    :param window_size: window size
    :param window_stride: window stride
    :param with_pca: apply pca
    :param pca_components: number of components to keep
    :param show_plot_variance: show the plot of the variance
    :param dataset_eval: evaluation dataset
    :param dataset_test: test dataset
    """
    dataset_train.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_train.remove_window(step=window_stride)

    dataset_eval.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_eval.remove_window(step=window_stride)

    dataset_test.add_window(type=window_type, window_size=window_size, stride=window_stride)
    dataset_test.remove_window(step=window_stride)

    dataset_train_process = copy.deepcopy(dataset_train)
    if dataset_eval is not None:
        dataset_eval_process = copy.deepcopy(dataset_eval)
    if dataset_test is not None:
        dataset_test_process = copy.deepcopy(dataset_test)

    # TRAININ SECTION
    dataset_train_process.add_window(
        type=window_type, window_size=window_size, stride=window_stride)

    scaler_model.fit(dataset_train_process.ts_data.data)
    dataset_train_process.normalize(normalizer=scaler_model, mode='complete')

    pca_model = None
    if with_pca:
        pca_model = fit_pca(dataset_train_process.ts_data.data, n_components=pca_components,
                    show_plot_variance=show_plot_variance)
        dataset_train_process.pca(model=pca_model, mode='complete')

    model = model.fit(dataset_train_process.ts_data.data)

    if dataset_eval is not None:
        dataset_eval_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_eval_process.normalize(normalizer=scaler_model, mode='single')

        # apply pca on eval dataset
        if with_pca:
            dataset_eval_process.pca(model=pca_model, mode='single')

        for ts_name, ts in dataset_eval_process.time_series.items():
            one_class_predict = model.predict(ts.data) == -1

            eval_anomalies_mask = np.array([one_class_predict]*6).T
            anomalies = copy.deepcopy(dataset_eval.time_series[ts_name].data)

            for i in range(len(one_class_predict)):
                if not one_class_predict[i]:
                    anomalies[i*window_size:i*window_size + window_size] = np.nan

            plot_ts(title=f"Local Outlier Factor {'PCA' if with_pca else 'NO PCA'} Figure Eval Anomalies {ts_name}", ts_name=f"eval_anomalies_{ts_name}", ts={'time_series': dataset_eval.time_series[ts_name].data, 'anomalies': anomalies}, features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'}, markers={'anomalies': 'o'}, save_plot=True, subfolder=f"local_outlier_{'pca' if with_pca else 'no_pca'}")


    # TEST SECTION
    if dataset_test is not None:
        dataset_test_process.add_window(
            type='tumbling', window_size=window_size)
        dataset_test_process.normalize(normalizer=scaler_model, mode='single')

        if with_pca:
            # apply pca on train dataset
            dataset_test_process.pca(model=pca_model, mode='single')

        tp = 0  # true positive, i.e. truth=anomaly, prediction=anomaly (anomaly detected correctly)
        fn = 0  # false negative, i.e. truth=anomaly, prediction=nominal (anomaly not detected)
        fp = 0  # false positive, i.e. truth=nominal, prediction=anomaly (nominal not detected)
        tn = 0  # true negative, i.e. truth=nominal, prediction=nominal (nominal detected correctly)

        for ts_name, ts in dataset_test_process.time_series.items():
            one_class_predict = model.predict(ts.data) == -1

            test_anomalies_mask = np.array([one_class_predict]*6).T
            anomalies = copy.deepcopy(dataset_test.time_series[ts_name].data)
            
            for i in range(len(one_class_predict)):
                if not one_class_predict[i]:
                    anomalies[i*window_size:i*window_size + window_size] = np.nan

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

            plot_ts(title=f"Local Outlier Factor {'PCA' if with_pca else 'NO PCA'} Figure Test Anomalies {ts_name}", ts_name=f"test_anomalies_{ts_name}", ts={'time_series': dataset_test.time_series[ts_name].data, 'anomalies': anomalies}, features=TimeSeriesUtils.PLOT_FEATURES,
                n_rows=2, n_cols=3, figsize=(15, 5), colors={'time_series': 'black', 'anomalies': 'red'}, markers={'anomalies': 'o'}, save_plot=True, subfolder=f"local_outlier_{'pca' if with_pca else 'no_pca'}")

        save_stats_txt(tp, fn, fp, tn, subfolder=f"local_outlier_{'pca' if with_pca else 'no_pca'}", filename=None, print_stats=True)
