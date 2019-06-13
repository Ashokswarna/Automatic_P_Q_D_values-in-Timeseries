"""
Created on 20 Mar 2018
Project: forecastlibrary
File: train
Author: prasenjit.giri
Copyright: Accenture AI
"""


from app_settings import read_config
from os import path
import pandas as pd
import pickle
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, _arma_predict_out_of_sample
from sklearn.metrics import mean_squared_error



train_period = [
    [201422, 201621],
    [201427, 201626],
    [201431, 201630],
    [201435, 201634],
    [201440, 201639]
]

validation_period = [
    [201622, 201630],
    [201627, 201634],
    [201631, 201639],
    [201635, 201643],
    [201640, 201647]
]

mse_period = [
    [201627, 201630],
    [201631, 201634],
    [201635, 201639],
    [201640, 201643],
    [201644, 201647]
]


p_range = range(0, 3)
d_range = range(0, 3)
q_range = range(0, 3)


def is_pvalue_significant(pvalues):
    result = True
    pvalues_counter = 0
    for pvalue in pvalues:
        if pvalues_counter != 0:
            if float(pvalue) > 0.05:
                result = False
                break
        pvalues_counter = pvalues_counter + 1

    return result


def revert_to_order(y_log, x_log, d_order=0):
    if d_order == 0:
        result = np.exp(y_log)
        return result
    else:
        pred_diff = pd.Series(y_log, copy=True)
        pred_diff_cumsum = pred_diff.cumsum()
        pred_log = pd.Series(x_log.iloc[0], index=x_log.index)
        pred_log = pred_log.add(pred_diff_cumsum, fill_value=0)
        result = np.exp(pred_log)
        return result


def mse_predictions(y_real, y_predicted):
    if len(y_real) > 4:
        y_predicted = y_predicted[-5:]
    else:
        y_predicted = y_predicted[-4:]

    print('Real:', y_real)
    print('Predicted:', y_predicted)
    mse = mean_squared_error(y_real, y_predicted)
    return mse


def evaluate_arima_model(x, y, order, start_period, end_period):
    x_log = transform_data(x)
    x_mat = x_log.as_matrix()

    y_mat = y.as_matrix()

    model = ARIMA(x_mat, order=order)
    model_fit = model.fit(disp=0)

    is_significant = is_pvalue_significant(model_fit.pvalues)
    print('model fit pvalues', model_fit.pvalues)

    if is_significant:
        print('Significant')

        # Keep a track of model parameters that are to be saved for predictions
        params = model_fit.params
        residuals = model_fit.resid
        p = model_fit.k_ar
        q = model_fit.k_ma
        k_exog = model_fit.k_exog
        k_trend = model_fit.k_trend
        intercept = params[0]

        # Predict values for the given steps
        y_predict_log = model_fit.predict(start=start_period, end=end_period, exog=None, dynamic=False)

        p_order, d_order, q_order = order

        # Revert predicted log values to normal scale
        y_predict_real = revert_to_order(y_predict_log, x_log, d_order)

        # select
        y_length = len(y)
        y_pred = y_predict_real[-y_length:]


        # Calculate the MSE for the last few predictions
        mse = mse_predictions(y_mat, y_pred)
    else:
        raise Exception('Insignificant model pvalues')

    return params, residuals, p, q, k_exog, k_trend, intercept, mse, y_predict_log


def evaluate_models(data, p_values, d_values, q_values):
    best_score = float('inf')
    best_cfg = None
    best_params = None
    best_residuals = None
    best_p = None
    best_q = None
    best_k_exog = None
    best_k_trend = None
    best_intercept = None

    # split data into train & validation test
    x, y, mse_sales = data

    # No of predictions includes Train + In-Time Validation weeks
    start_step = 1
    # end_step = 137
    # if len(y) > 4:
    #     end_step = 138

    end_step = len(x) + len(y)


    for p_value in p_values:
        for d_value in d_values:
            for q_value in q_values:
                # split data
                try:
                    params, residuals, p, q, k_exog, k_trend, intercept, mse, y_predict_log = evaluate_arima_model(x, mse_sales,
                                                                                                                   order=(
                                                                                                                       p_value,
                                                                                                                       d_value,
                                                                                                                       q_value),
                                                                                                                   start_period=start_step,
                                                                                                                   end_period=end_step)
                    print('Order p:%d d:%d q:%d' % (p_value, d_value, q_value))
                    # print(params, residuals, p, q, k_exog, k_trend, intercept)
                    print(params, p, q, k_exog, k_trend, intercept)
                    print('MSE:', mse)

                    # Keep track of best least mse model parameters
                    if mse < best_score:
                        best_score = mse
                        best_cfg = (p_value, d_value, q_value)
                        best_params = params
                        best_residuals = residuals
                        best_p = p
                        best_q = q
                        best_k_exog = k_exog
                        best_k_trend = k_trend
                        best_intercept = intercept

                    print()
                except Exception as e:
                    print('Order p:%d d:%d q:%d' % (p_value, d_value, q_value))
                    print('Failed')
                    print(e)
                    print()
                    pass

    return (
    best_score, best_cfg, best_params, best_residuals, best_p, best_q, best_k_exog, best_k_trend, best_intercept,
    y_predict_log)


def transform_data(data):
    data_log = np.log(data)
    data_log[data_log == -np.inf] = 0
    data_log[data_log == np.inf] = 0
    return data_log



def find_best_model(models):
    print('BEGIN: Selection of best model across periods')

    best_model_score = float('inf')
    best_model = {}
    for model in models:
        if model['mse'] < best_model_score:
            best_model = model
            best_model_score = model['mse']

    # print(best_model)
    # Find best model on R2
    # best_ols_model = find_best_ols_model(models)
    # best_model['best_ols_params'] = best_ols_model

    print()
    print('END: Selection of best model across weeks')

    return best_model


def save_model_to_disk(file_name, model):
    pickle.dump(model, open(file_name, 'wb'))
    print('%s saved' % file_name)



def train(filename):
    """
        Trains ARIMA model post least MSE per sku & selects the best model and saves it
    :return: None
    """

    app_settings = read_config()
    data_path = app_settings['data_path']
    file_path = path.join(data_path, filename)
    print(file_path)

    begin = 0
    end = 1

    df = pd.read_csv(file_path)

    # Columns: Sku, Week, Sales

    sku_group = df.groupby('Sku', as_index=False)
    sku_list = sku_group.groups.keys()

    sku_best_model = []

    for sku in sku_list:
        print()
        print(sku)

        # Select SKU to train & validate model
        df_sku = df[df['Sku'].isin([sku])]
        period_index = 0
        best_period_models = []

        for tp in train_period:
            print()
            print('Begin:%d End:%d' % (tp[0], tp[1]))
            print()
            # Select SKU data from beginning to end of train period
            df_train_period = df_sku[
                (df_sku['Week'] >= tp[begin]) & (df_sku['Week'] <= tp[end])
                ]

            # Select SKU data from beginning to end of in-time validation period
            df_validation_period = df_sku[
                (df_sku['Week'] >= validation_period[period_index][begin]) & (
                        df_sku['Week'] <= validation_period[period_index][end])
                ]

            df_mse_period = df_sku[
                (df_sku['Week'] >= mse_period[period_index][begin]) & (
                        df_sku['Week'] <= mse_period[period_index][end])
                ]

            print('%d train samples for %d period.' % (len(df_train_period), (period_index + 1)))
            print('%d validation samples for %d period.' % (len(df_validation_period), (period_index + 1)))
            print('%d mse samples for %d period.' % (len(df_mse_period), (period_index + 1)))

            # Select sales data for training & validation
            train_sales = df_train_period['Sales'].reset_index(drop=True)
            validation_sales = df_validation_period['Sales'].reset_index(drop=True)
            mse_sales = df_mse_period['Sales'].reset_index(drop=True)

            train_valid_set = (train_sales, validation_sales, mse_sales)

            # Evaluate best model of selected train period
            best_score, best_cfg, best_params, best_residuals, best_p, best_q, best_k_exog, best_k_trend, best_intercept, y_predict_log = evaluate_models(
                train_valid_set, p_range, d_range, q_range)

            best_period_model = {'best_cfg': best_cfg, 'mse': best_score, 'sku': sku, 'week': (period_index + 1),
                                 'residuals': best_residuals, 'p': best_p, 'q': best_q, 'k_exog': best_k_exog,
                                 'k_trend': best_k_trend,
                                 'params': best_params, 'intercept': best_intercept}
            best_period_models.append(best_period_model)
            period_index += 1

        # Select best model in entire period
        best_model = find_best_model(best_period_models)

        # Add to best models list
        sku_best_model.append(best_model)
        print('____________________________________________________________________________________________')
        print('____________________________________________________________________________________________')

    # Save model to disk
    model_path = app_settings['model_path']

    file_parts = filename.split('.')
    # model_file_name = file_parts[0] + '_HyperParameters.pickle'
    model_file_name = 'model.pickle'

    model_file_path = path.join(model_path, model_file_name)
    save_model_to_disk(model_file_path, sku_best_model)

    print('Training completed')



def main():
    train('demo_train.csv')


if __name__ == '__main__':
    main()

