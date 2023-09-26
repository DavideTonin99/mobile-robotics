from classes.MainLocalOutlierFactor import MainLocalOutlierFactor
from classes.MainNNLinearRegression import MainNNLinearRegression
from classes.MainSVM import MainSVM
from classes.Params import Params
from classes.MainPCA import MainPCA
from sklearn.preprocessing import StandardScaler

train_list = [f'nominal_{i}' for i in range(7, 8)]
eval_list = [f'nominal_{i}' for i in range(1, 7)]
test_list = [f'anomaly_{i}' for i in range(1, 11)]
test_list += eval_list

params = Params({
    'APPLY_MOVING_AVG': True,
    'MOVING_AVG_STEP': 10,
    'WINDOW_TYPE': 'sliding',
    'WINDOW_SIZE': 15,
    'WINDOW_STRIDE': 5,
    'APPLY_PCA': True,
    'PCA_COMPONENTS': 7,
    'NORMALIZER_MODEL': StandardScaler(),
    'THRESHOLD_TYPE': 'svm',
    'QUANTILE_LOWER_PERCENTAGE': 0.01,
    'QUANTILE_UPPER_PERCENTAGE': 0.99,
})
"""
print("Exec PCA with svm")
main = MainPCA(params=params)
main.run(train_list=train_list, eval_list=[], test_list=test_list, show_plot=True)

print("Exec PCA with local outlier factor")
params.THRESHOLD_TYPE = 'local_outlier_factor'
main = MainPCA(params=params)
main.run(train_list=train_list, eval_list=[], test_list=test_list, show_plot=True)

print("Exec PCA with quantile")
params.THRESHOLD_TYPE = 'quantile'
main = MainPCA(params=params)
main.run(train_list=train_list, eval_list=[], test_list=test_list, show_plot=True)

print("Exec NN Linear Regression")
main = MainNNLinearRegression(params=params)
main.run(train_list=train_list, eval_list=eval_list, test_list=test_list, show_plot=True)

print("Exec Local Outlier Factor")
main = MainLocalOutlierFactor(params=params)
main.run(train_list=train_list, eval_list=[], test_list=test_list, show_plot=True)
"""
print("Exec SVM")
train_list_svm = train_list + [f'anomaly_{i}' for i in range(10, 11)]
main = MainSVM(params=params)
main.run(train_list=train_list_svm, eval_list=[], test_list=test_list, show_plot=True)
