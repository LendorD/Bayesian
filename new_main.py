import os
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from joblib import dump
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import time
from pathlib import Path
import sys

sys.path.append('/home/ubuntu/python/miscellaneous_library')
# import confusion_matrix_plot


# *******************************************************************

def create_save_models_bayes_opt(x, y, error_metric, cv_folds,
                                 num_random_points, num_iterations, results_dir):
    start_time_total = time.time()

    def etrees_crossval(n_estimators, max_features, min_samples_split,
                        min_samples_leaf, max_samples):
        etrees = ExtraTreesClassifier(n_estimators=int(n_estimators),
                                      max_features=max_features, min_samples_split=int(min_samples_split),
                                      min_samples_leaf=int(min_samples_leaf), max_samples=max_samples,
                                      bootstrap=True, n_jobs=-1, verbose=0)

        mean_cv_score = cross_val_score(etrees, x, y, scoring=error_metric,
                                        cv=cv_folds, n_jobs=-1).mean()

        return mean_cv_score

    optimizer = BayesianOptimization(f=etrees_crossval,
                                     pbounds={'n_estimators': (25, 251),
                                              'max_features': (0.15, 1.0),
                                              'min_samples_split': (2, 14),
                                              'min_samples_leaf': (1, 14),
                                              'max_samples': (0.6, 0.99)},
                                     verbose=2)

    optimizer.maximize(init_points=num_random_points,
                       n_iter=num_iterations)

    print('nbest result:', optimizer.max)

    elapsed_time_total = (time.time() - start_time_total) / 60
    print('\n\ntotal elapsed time =', elapsed_time_total, ' minutes')

    # optimizer.res is a list of dicts
    list_dfs = []
    counter = 0
    for result in optimizer.res:
        df_temp = pd.DataFrame.from_dict(data=result['params'], orient='index',
                                         columns=['trial' + str(counter)]).T
        df_temp[error_metric] = result['target']

        list_dfs.append(df_temp)

        counter = counter + 1

    df_results = pd.concat(list_dfs, axis=0)
    df_results.to_pickle(results_dir + 'df_bayes_opt_results_parameters.pkl')
    df_results.to_csv(results_dir + 'df_bayes_opt_results_parameters.csv')


# end of create_save_models_bayes_opt()

# *******************************************************************

def make_final_predictions(xcalib, ycalib, xprod, yprod,
                           list_class_names, models_directory,
                           save_directory, save_models_flag, df_params,
                           threshold, type_error, ml_name):
    # apply threshold
    accepted_models_num = 0
    list_predicted_prob = []
    num_models = df_params.shape[0]
    for i in range(num_models):
        if df_params.loc[df_params.index[i], type_error] > threshold:
            ml_model = ExtraTreesClassifier(
                n_estimators=int(df_params.loc[df_params.index[i], 'n_estimators']),
                max_features=df_params.loc[df_params.index[i], 'max_features'],
                min_samples_split=int(df_params.loc[df_params.index[i], 'min_samples_split']),
                min_samples_leaf=int(df_params.loc[df_params.index[i], 'min_samples_leaf']),
                max_samples=df_params.loc[df_params.index[i], 'max_samples'],
                bootstrap=True, n_jobs=-1, verbose=0)

            ml_model.fit(xcalib, ycalib)

            list_predicted_prob.append(ml_model.predict_proba(xprod))

            accepted_models_num = accepted_models_num + 1

            if save_models_flag:
                model_name = ml_name + df_params.index[i] + '_joblib.sav'
                dump(ml_model, save_directory + model_name)

    # compute mean probabilities
    mean_probabilities = np.mean(list_predicted_prob, axis=0)

    # compute predicted class
    # argmax uses 1st ocurrance in case of a tie
    y_predicted_class = np.argmax(mean_probabilities, axis=1)

    # compute and save error measures

    # print info to file
    stdout_default = sys.stdout
    sys.stdout = open(save_directory + ml_name + '_prediction_results.txt', 'w')

    print('balanced accuracy score =', balanced_accuracy_score(yprod, y_predicted_class))

    print('accuracy score =', accuracy_score(yprod, y_predicted_class))

    print('number of accepted models =', accepted_models_num, ' for threshold =', threshold)

    print('\nclassification report:')
    print(classification_report(yprod, y_predicted_class, digits=3, output_dict=False))

    print('\nraw confusion matrix:')
    cm_raw = confusion_matrix(yprod, y_predicted_class)
    print(cm_raw)

    print('\nconfusion matrix normalized by prediction:')
    cm_pred = confusion_matrix(yprod, y_predicted_class, normalize='pred')
    print(cm_pred)

    print('\nconfusion matrix normalized by true:')
    cm_true = confusion_matrix(yprod, y_predicted_class, normalize='true')
    print(cm_true)

    sys.stdout = stdout_default

    # plot and save confustion matrices
    figure_size = (12, 8)
    number_of_decimals = 4
    #
    # confusion_matrix_plot.confusion_matrix_save_and_plot(cm_raw,
    #                                                      list_class_names, save_directory, 'Confusion Matrix',
    #                                                      ml_name + '_confusion_matrix', False, None, 30, figure_size,
    #                                                      number_of_decimals)
    #
    # confusion_matrix_plot.confusion_matrix_save_and_plot(cm_pred,
    #                                                      list_class_names, save_directory,
    #                                                      'Confusion Matrix Normalized by Prediction',
    #                                                      ml_name + '_confusion_matrix_norm_by_prediction', False,
    #                                                      'pred',
    #                                                      30, figure_size, number_of_decimals)
    #
    # confusion_matrix_plot.confusion_matrix_save_and_plot(cm_true,
    #                                                      list_class_names, save_directory,
    #                                                      'Confusion Matrix Normalized by Actual',
    #                                                      ml_name + '_confusion_matrix_norm_by_true', False, 'true',
    #                                                      30, figure_size, number_of_decimals)


# end of make_final_predictions()

# *******************************************************************

if __name__ == '__main__':

    ml_algorithm_name = 'etrees'
    file_name_stub = ml_algorithm_name + '_bayes_opt'

    calculation_type = 'production'  # 'calibration' 'production'

    data_directory = 'C:/Users/Lendor/PycharmProjects/Bayesian'

    base_directory = 'C:/Users/Lendor/PycharmProjects/Bayesian'

    results_directory_stub = base_directory + file_name_stub + '/'
    if not Path(results_directory_stub).is_dir():
        os.mkdir(results_directory_stub)

    # fixed parameters
    error_type = 'balanced_accuracy'
    threshold_error = 0.93
    cross_valid_folds = 3
    total_number_of_iterations = 50
    number_of_random_points = 10  # random searches to start opt process
    # this is # of bayes iters, thus total=this + # of random pts
    number_of_iterations = total_number_of_iterations - number_of_random_points
    save_models = False

    # use small data set
    x_calib = np.load(data_directory + 'x_mnist_calibration_1.npy')
    y_calib = np.load(data_directory + 'y_mnist_calibration_1.npy')

    print('\n*** starting at', pd.Timestamp.now())

    # 1 - calibration - using cross validation, get cv score for the given set of
    # parameters via bayes opt search
    if calculation_type == 'calibration':

        results_directory = results_directory_stub + calculation_type + '/'
        if not Path(results_directory).is_dir():
            os.mkdir(results_directory)

        create_save_models_bayes_opt(x_calib, y_calib, error_type,
                                     cross_valid_folds,
                                     number_of_random_points, number_of_iterations,
                                     results_directory)

    # 2 - production - apply threshold
    elif calculation_type == 'production':

        # get etrees parameters
        models_dir = results_directory_stub + 'calibration/'
        df_parameters = pd.read_pickle(models_dir + 'df_bayes_opt_results_parameters.pkl')

        results_directory = results_directory_stub + calculation_type + '/'
        if not Path(results_directory).is_dir():
            os.mkdir(results_directory)

        x_prod = np.load(data_directory + 'x_mnist_production.npy')
        y_prod = np.load(data_directory + 'y_mnist_production.npy')

        num_classes = np.unique(y_prod).shape[0]
        class_names_list = []
        for i in range(num_classes):
            class_names_list.append('class ' + str(i))

        make_final_predictions(x_calib, y_calib, x_prod, y_prod,
                               class_names_list,
                               models_dir, results_directory,
                               save_models, df_parameters,
                               threshold_error, error_type, ml_algorithm_name)

    else:
        print('\ninvalid calculation type:', calculation_type)
        raise NameError