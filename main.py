from TimeSeriesUtils import *
from TimeSeries import *
from utils import *
import copy
from NNLinearRegression import *


def main_pca(dataset_train, scaler_model, window_type, window_size, window_stride, pca_components, show_plot_variance=True, dataset_eval=None, dataset_test=None):
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
    # errors = dataset_train_process.ts_data - dataset_train.ts_data

    plot_ts(title="Figure Train", ts_name="train", ts={'start': dataset_train.ts_data.data, 'end': dataset_train_process.ts_data.data}, features=TimeSeriesUtils.PLOT_FEATURES,
            n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True)

    if dataset_eval is not None:
        dataset_eval_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_eval_process.normalize(normalizer=scaler_model, mode='single')

        # apply pca on train dataset
        dataset_eval_process.pca(model=model, mode='single')
        dataset_eval_process.pca_inverse(model=model, mode='single')

        # back to original scale
        dataset_eval_process.normalize_inverse(
            normalizer=scaler_model, mode='single')
        dataset_eval_process.remove_window(step=window_stride)

        for ts_name, ts in dataset_eval_process.time_series.items():
            plot_ts(title=f"Figure Eval {ts_name}", ts_name=ts_name, ts={'start': dataset_eval.time_series[ts_name].data, 'end': dataset_eval_process.time_series[ts_name].data}, features=TimeSeriesUtils.PLOT_FEATURES,
                    n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True)

    # TEST SECTION
    if dataset_test is not None:
        dataset_test_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_test_process.normalize(normalizer=scaler_model, mode='single')

        # apply pca on train dataset
        dataset_test_process.pca(model=model, mode='single')
        dataset_test_process.pca_inverse(model=model, mode='single')

        # back to original scale
        dataset_test_process.normalize_inverse(
            normalizer=scaler_model, mode='single')
        dataset_test_process.remove_window(step=window_stride)

        for ts_name, ts in dataset_test_process.time_series.items():
            plot_ts(title=f"Figure Test {ts_name}", ts_name=ts_name, ts={'start': dataset_test.time_series[ts_name].data, 'end': dataset_test_process.time_series[ts_name].data}, features=TimeSeriesUtils.PLOT_FEATURES,
                    n_rows=2, n_cols=3, figsize=(15, 5), colors={'start': 'black', 'end': 'orange'}, save_plot=True)


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

    q1, q2 = np.quantile(loss_log['t'], [0.02, 0.98], axis=0)
    pc_iqr = q2 - q1
    pc_lower_bound = q1 - (1 * pc_iqr)
    pc_upper_bound = q2 + (1 * pc_iqr)

    print(f"* Lower Bound: {pc_lower_bound}")
    print(f"* Upper Bound: {pc_upper_bound}")

    if dataset_test is not None:
        dataset_test_process.add_window(
            type=window_type, window_size=window_size, stride=window_stride)
        dataset_test_process.normalize(normalizer=scaler_model, mode='single')

        for t_name, t in dataset_test_process.time_series.items():
            is_anomaly_gt = 'anomaly' in t_name
            is_anomaly_pred = False
            count_windows_anomaly = 0
            errors_list = []

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
                    is_anomaly_pred = True
                    errors_list.append(error)
                
            print(f"{t_name}: Anomalies detected: {count_windows_anomaly}, {errors_list}")