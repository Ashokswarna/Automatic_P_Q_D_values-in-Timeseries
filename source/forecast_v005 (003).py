import warnings
import numpy as np
import pandas as pd
import os
import pickle
import statsmodels.api as sm
import copy
from statsmodels.tsa.arima_model import ARIMA, _arma_predict_out_of_sample
from sklearn.metrics import mean_squared_error
from datetime import datetime

warnings.filterwarnings('ignore')



def generate_data(file_name):
    os.chdir(data_path)
    df = pd.read_csv(file_name)
    df = df.groupby(['ForecastUnitCode', 'ForecastWeek'], as_index=False)['PrimaryDeliveries1'].sum()
    df.rename(columns={'PrimaryDeliveries1': 'Weekly_Volume_Sales'}, inplace=True)
    df.fillna(0, inplace=True)

    df.to_csv('Tea_Data_Set_Full.csv')
    print('Saved file')


def calc_seasonal_index(group):
    weeks = range(1, 53)
    years = range(2014, 2017)
    dict = {}

    # for index, row in group.iterrows():
    for week in weeks:

        # find all data for relevant weeks
        weekly_sales = group[group['Week'] == str(week).zfill(2)]
        first_year_week_ratio = weekly_sales[(weekly_sales['Year'] == '2014' )]['Ratio_MA']
        second_year_week_ratio = weekly_sales[(weekly_sales['Year'] == '2015')]['Ratio_MA']
        third_year_week_ratio = weekly_sales[(weekly_sales['Year'] == '2016')]['Ratio_MA']

        if not first_year_week_ratio.empty:
            df_concat = pd.concat((first_year_week_ratio, second_year_week_ratio, third_year_week_ratio))

            avg = df_concat.mean()
        else:
            df_concat = pd.concat(( second_year_week_ratio, third_year_week_ratio))
            avg = df_concat.mean()

        dict[str(week).zfill(2)] = avg

    seasonal_group = group.copy()
    seasonal_group['Seasonal_Index'] = seasonal_group['Week'].replace(dict)
    return  seasonal_group


def deseasonalize_sales(filename):

    os.chdir(data_path)
    df = pd.read_csv(filename)

    seasonal_groups = []

    sku_group = df.groupby('ForecastUnitCode', as_index=False)
    sku_list = sku_group.groups.keys()

    for sku in sku_list:
        df_sku = df[df['ForecastUnitCode'] == '3100:FGB0762']
        # Select complete period for training



    df = df[(df['ForecastWeek'] >= 201422) & (df['ForecastWeek'] <=201716)]
    df = df[df['ForecastUnitCode'] =='3100:FGB0762']
    df['4_Wk_MA'] = df['Weekly_Volume_Sales'].rolling(window=4).mean()
    df['Ratio_MA'] = df['Weekly_Volume_Sales'] / df['4_Wk_MA']
    df['Week'] = df['ForecastWeek'].astype(str).str[-2:]
    df['Year'] = df['ForecastWeek'].astype(str).str[:4]
    # df['Seasonal_Index'] = df['Ratio_MA'].apply(lambda x: )
    # df.apply(calc_seasonal_index)
    calc_seasonal_index(df)
    print(df.head())


def split_seq(seq, num_pieces):
    newseq = []
    split_size = 1.0 / num_pieces * len(seq)
    for i in range(num_pieces):
        newseq.append(seq[int(round(i * split_size)):int(round((i + 1) * split_size))])
    return newseq


def save_group(data_frame, groups, filename='Tea_Data_Set_'):
    os.chdir(data_path)
    group_count = 1
    for group in groups:
        new_df = data_frame[data_frame.ForecastUnitCode.isin(group)]
        file_name = filename + str(group_count) + '.csv'
        new_df.to_csv(file_name)
        print('%s saved' % file_name)
        group_count = group_count + 1


def create_file_name(filename='Result', extension='xlsx'):
    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime('%d_%m_%Y_%H_%M_%S')
    filename = filename + "_" + str_current_datetime + "." + extension
    return filename


def split_data_into_smaller_files(file_name):
    os.chdir(data_path)
    df = pd.read_csv(file_name)
    df = df.groupby(['ForecastUnitCode', 'ForecastWeek'], as_index=False)['PrimaryDeliveries1'].sum()
    df.rename(columns={'PrimaryDeliveries1': 'Weekly_Volume_Sales'}, inplace=True)
    df.fillna(0, inplace=True)
    total_sku = df.ForecastUnitCode.unique()
    split_sku = split_seq(total_sku, 3)
    save_group(df, split_sku)


def transform_data(data):
    data_log = np.log(data)
    data_log[data_log == -np.inf] = 0
    data_log[data_log == np.inf] = 0
    return data_log


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


def find_best_ols_model(models):
    max_r2_score = 0
    best_ols_model_param = {}
    for model in models:
        if model['r2_score'] > max_r2_score:
            max_r2_score = model['r2_score']
            best_ols_model_param['pvalues'] = model['promo_sig']
            best_ols_model_param['max_r2'] = model['r2_score']

    return best_ols_model_param


def save_model_to_disk(file_name, model):
    pickle.dump(model, open(file_name, 'wb'))
    print('%s saved' % file_name)


def rename_file(filename, part, ext):
    file = filename.split('.')
    new_file_name = file[0] + '_' + part + '.' + ext
    return new_file_name


def train():
    """
        Trains ARIMA model post least MSE per sku & selects the best model and saves it
    :return: None
    """

    for filename in sales_files:
        # Change path to data folder to read file
        os.chdir(data_path)
        df = pd.read_csv(filename)

        # Logic test across few sku
        # df = df[df.ForecastUnitCode.isin(['3100:FGB0737', '3100: FGB0723', '3100: FGB6542'])]

        sku_group = df.groupby('ForecastUnitCode', as_index=False)
        sku_list = sku_group.groups.keys()

        sku_best_model = []

        for sku in sku_list:
            print()
            print(sku)

            # Select SKU to train & validate model
            df_sku = df[df.ForecastUnitCode.isin([sku])]
            period_index = 0
            best_period_models = []

            for tp in train_period:
                print()
                print('Begin:%d End:%d' % (tp[0], tp[1]))
                print()
                # Select SKU data from beginning to end of train period
                df_train_period = df_sku[
                    (df_sku.ForecastWeek >= tp[begin]) & (df_sku.ForecastWeek <= tp[end])
                    ]

                # Select SKU data from beginning to end of validation period
                df_validation_period = df_sku[
                    (df_sku.ForecastWeek >= validation_period[period_index][begin]) & (
                            df_sku.ForecastWeek <= validation_period[period_index][end])
                    ]

                df_mse_period = df_sku[
                    (df_sku.ForecastWeek >= mse_period[period_index][begin]) & (
                            df_sku.ForecastWeek <= mse_period[period_index][end])
                    ]


                print('%d train samples for %d period.' % (len(df_train_period), (period_index + 1)))
                print('%d validation samples for %d period.' % (len(df_validation_period), (period_index + 1)))
                print('%d mse samples for %d period.' % (len(df_mse_period), (period_index + 1)))

                # Select sales data for training & validation
                train_sales = df_train_period['Weekly_Volume_Sales'].reset_index(drop=True)
                validation_sales = df_validation_period['Weekly_Volume_Sales'].reset_index(drop=True)
                mse_sales = df_mse_period['Weekly_Volume_Sales'].reset_index(drop=True)

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
        os.chdir(result_path)
        file = filename.split('.')
        new_file_name = file[0] + '_HyperParameters.pickle'
        save_model_to_disk(new_file_name, sku_best_model)

    print('Training completed')


def forecast_oot():
    # Set of 8 weeks for forecasting out of time samples
    # test_period = [[201706, 201713],
    #                [201710, 201717],
    #                [201714, 201721]
    #                ]

    test_period = [
        [201636, 201643],
        [201640, 201647],
        [201645, 201652],
        [201649, 201704],
        [201701, 201708]
    ]

    # Out of time period
    # oot_period = [[201714, 201717],
    #               [201718, 201721],
    #               [201722, 201726]
    #               ]

    oot_period = [
        [201648, 201652],
        [201701, 201704],
        [201705, 201708],
        [201709, 201713],
        [201714, 201717]
    ]

    for filename in sales_files:

        # Load the model hyperparameters for file
        model_file_name = rename_file(filename, 'HyperParameters', 'pickle')
        os.chdir(result_path)
        model_params = pickle.load(open(model_file_name, 'rb'))

        # Read sku-sales data for forecasting
        os.chdir(data_path)
        df = pd.read_csv(filename)

        # df = df[df.ForecastUnitCode.isin(['3100:FGB0723'])]
        # df = df[df.ForecastUnitCode.isin(['3100:FGB0737', '3100: FGB0723', '3100: FGB6542'])]

        sku_group = df.groupby('ForecastUnitCode', as_index=False)
        sku_list = sku_group.groups.keys()

        total_predictions = []

        for sku in sku_list:
            df_sku = df[df.ForecastUnitCode.isin([sku])]
            period_index = 0
            print('-----------------------------------------------------')
            print('Result for SKU:', sku)
            for period in test_period:

                x_train = df_sku[
                    (df_sku.ForecastWeek >= period[0]) &
                    (df_sku.ForecastWeek <= period[1])
                    ]
                x_train = x_train['Weekly_Volume_Sales'].reset_index(drop=True)
                x_log = transform_data(x_train)
                history = [x for x in x_log]

                # y_test = df_sku[
                #     (df_sku.ForecastWeek >= oot_period[period_index][0]) &
                #     (df_sku.ForecastWeek <= oot_period[period_index][1])
                #     ]
                # y_test = y_test['Weekly_Volume_Sales'].reset_index(drop=True)

                for model_param in model_params:
                    if model_param['sku'] == sku:
                        p_order, d_order, q_order = model_param['best_cfg']

                        if d_order > 0:
                            print('Difference SKU %s with order %d' % (sku, d_order))
                            # No second order differencing exists in our model, hence only 1st order is required
                            history = difference(history)

                        print('week:', period_index)
                        params = model_param['params']
                        residuals = model_param['residuals']
                        p = model_param['p']
                        q = model_param['q']
                        k_exog = model_param['k_exog']
                        k_trend = model_param['k_trend']
                        intercept = model_param['intercept']
                        # steps = 4

                        y_real = df_sku[
                            (df_sku.ForecastWeek >= oot_period[period_index][0]) & (
                                    df_sku.ForecastWeek <= oot_period[period_index][1])].reset_index(
                            drop=True)

                        # if len(y_real) > 4:
                        #     steps = 5

                        steps = len(y_real)
                        # print('intercept %d' % intercept)

                        y_predicted_log = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog,
                                                                      endog=history, exog=None, start=len(history))

                        y_predicted = revert_to_order(y_predicted_log, x_log, d_order)
                        y_pred_series = pd.Series(y_predicted)

                        y_real.drop(y_real.columns[[0]], axis=1, inplace=True)
                        y_real['Predicted_Weekly_Volume_Sales'] = y_pred_series
                        # print(y_real)
                        # print()
                        total_predictions.append(y_real)

                period_index += 1

        # Save predicted sales for respective files
        result_file_name = rename_file(filename, 'Result', 'csv')

        result_df = pd.concat(total_predictions)
        result_df.reset_index(drop=True, inplace=True)

        # Change path to result folder
        os.chdir(result_path)
        result_df.to_csv(result_file_name, sep=',')
        print('Forecasting completed for %s' % filename)


def filter_by_sku():
    result_file_names = ['Tea_Data_Set_1_Result.csv', 'Tea_Data_Set_2_Result.csv', 'Tea_Data_Set_3_Result.csv']

    ref_file_name = 'Tea_FU_APG_WK_Forecast_results_Dec16_Apr17_1_23.csv'
    os.chdir(data_path)
    df_ref = pd.read_csv(ref_file_name)

    ref_sku_group = df_ref.groupby('ForecastUnitCode', as_index=False)
    ref_sku_list = ref_sku_group.groups.keys()


    res_file_name = "Filtered_SKU_Forecast_Results.csv"

    os.chdir(result_path)
    filtered_df = []
    for file in result_file_names:
        df = pd.read_csv(file)
        df_filtered = df[df['ForecastUnitCode'].isin(ref_sku_list)]
        filtered_df.append(df_filtered)

    # Create a dataframe
    result_df = pd.concat(filtered_df)
    result_df.reset_index(drop=True, inplace=True)

    # Change path to result folder
    os.chdir(result_path)
    result_df.to_csv(res_file_name, sep=',')
    print('Forecasting completed for %s' % res_file_name)


def filter_record():
    result_file_names = ['Tea_Data_Set_1_Result.csv', 'Tea_Data_Set_2_Result.csv', 'Tea_Data_Set_3_Result.csv']

    sku_with_sales = []
    os.chdir(result_path)

    for filename in result_file_names:
        df = pd.read_csv(filename)
        sku_group = df.groupby('ForecastUnitCode', as_index=False)
        sku_list = sku_group.groups.keys()

        zero_sales = 0

        for sku in sku_list:
            df_sku = df[df.ForecastUnitCode.isin([sku])]
            df_sales = df_sku.loc[df_sku['Weekly_Volume_Sales'] != zero_sales]
            if (len(df_sku) == len(df_sales)):
                sku_with_sales.append(df_sku)

    result_df = pd.concat(sku_with_sales)
    result_df.reset_index(drop=True, inplace=True)

    # Change path to result folder
    os.chdir(result_path)
    result_df.to_csv('Tea_Results_SKU_With_Sales.csv', sep=',')


def find_sku_for_difference():
    os.chdir(result_path)
    model__files = ['Tea_Data_Set_1_HyperParameters.pickle', 'Tea_Data_Set_2_HyperParameters.pickle',
                    'Tea_Data_Set_3_HyperParameters.pickle']

    sku_diff = []

    for model_file in model__files:
        # Unpack pickle file
        models = pickle.load(open(model_file, 'rb'))

        for model in models:
            p_order, d_order, q_order = model['best_cfg']
            if d_order > 0:
                result = {'ForecastUnitCode': model['sku'],
                          'p': p_order,
                          'd': d_order,
                          'q': q_order
                          }
                sku_diff.append(result)

    result_df = pd.DataFrame(sku_diff)
    result_df.reset_index(drop=True, inplace=True)

    # Change path to result folder
    os.chdir(result_path)
    result_df.to_csv('Tea_Results_SKU_With_Diff.csv', sep=',')


def calc_delta(filenames=['Tea_Results_SKU_With_Sales.csv']):
    for filename in filenames:
        os.chdir(result_path)
        df = pd.read_csv(filename)
        sku_group = df.groupby('ForecastUnitCode', as_index=False)
        sku_list = sku_group.groups.keys()

        result = []
        for sku in sku_list:
            df_sku = df[df.ForecastUnitCode.isin([sku])]

            #
            df_sku['Prev_WSD'] = df_sku['Weekly_Volume_Sales'].shift().fillna(0)
            # Previous - Current
            df_sku['Prev_Current_Diff'] = (df_sku['Prev_WSD'] - df_sku['Weekly_Volume_Sales']).abs()
            # (Previous-Current) / Previous
            df_sku['Delta'] = df_sku['Prev_Current_Diff'] / df_sku['Prev_WSD']

            df_sku['Weekly_Sales_Delta'] = np.where(df_sku['Delta'] == np.inf, df_sku['Weekly_Volume_Sales'],
                                                    df_sku['Delta'])

            # df_sku['Weekly_Sales_Delta'] = ( (df_sku['Weekly_Volume_Sales'].shift() - df_sku['Weekly_Volume_Sales']) /
            #                                  (df_sku['Weekly_Volume_Sales'].shift().apply(lambda x: 1 if x<0 else x))
            #                                  ).fillna(0).abs()

            df_sku['Weekly_Sales_Delta_Perc'] = df_sku['Weekly_Sales_Delta'] * 100
            # df_sku['Weekly_Sales_Delta_Perc'] = df_sku['Weekly_Sales_Delta_Perc'].round(2)
            result.append(df_sku)

        df_result = pd.concat(result)
        df_result.reset_index(drop=True, inplace=True)

        columns = ['ForecastUnitCode', 'ForecastWeek', 'Weekly_Volume_Sales', 'Predicted_Weekly_Volume_Sales',
                   'Weekly_Sales_Delta', 'Weekly_Sales_Delta_Perc']

        df_result = pd.DataFrame(df_result, columns=columns)
        result_file_name = filename.split('.')[0] + '_Delta.csv'
        df_result.to_csv(result_file_name)

    print('Sales delta completed')


def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def generate_results_for(filename):
    result_file_names = ['Tea_Data_Set_1_Result.csv', 'Tea_Data_Set_2_Result.csv', 'Tea_Data_Set_3_Result.csv']

    os.chdir(data_path)
    df = pd.read_excel(filename)
    sku_list = np.array(df.iloc[:, 0])

    os.chdir(result_path)
    result = []
    for filename in result_file_names:
        df_res = pd.read_csv(filename)
        df_res = df_res[df_res.ForecastUnitCode.isin(sku_list)]
        result.append(df_res)

    result_df = pd.concat(result)
    result_df.reset_index(drop=True, inplace=True)
    result_df.to_csv('Result_SKUs_Ops.csv')
    print('Result saved for SKUs_Ops')


def normalize_promo(value):
    result = 0

    if value > 0:
        result = 1
    return result


def prepare_data(file_name):
    """

    :param file_name:
    :return:
    """
    os.chdir(data_path)
    df = pd.read_csv(file_name)
    cols = ['PM_Pipefill_Plinth', 'PM_Special_Pack_Extra_Free', 'PM_Multi_Buy', 'PM_Price_Reduction',
            'PM_Special_Pack_Other', 'PM_Other_Mechanics', 'PM_Price_Reduction_Multi_Buy',
            'PM_Special_Pack_Price_Marked_Pack']

    all_cols = copy.copy(cols)
    all_cols.append('PrimaryDeliveries1')
    df = df.groupby(['ForecastUnitCode', 'ForecastWeek'], as_index=False)[all_cols].sum()

    for col in cols:
        df[col] = df[col].apply(lambda x: normalize_promo(x))
    # df[cols].apply(lambda x: normalize_promo(x))
    df.rename(columns={'PrimaryDeliveries1': 'Weekly_Volume_Sales'}, inplace=True)
    df.fillna(0, inplace=True)
    total_sku = df.ForecastUnitCode.unique()
    split_sku = split_seq(total_sku, 3)
    save_group(df, split_sku, filename='Tea_Data_Set_Promo_')


def predict_with_residues(filenames):
    # Set of 8 weeks feed into ARIMA for forecasting out of time samples
    test_period = [[201706, 201713],
                   [201710, 201717],
                   [201714, 201721]
                   ]

    oot_period = [[201714, 201717],
                  [201718, 201721],
                  [201722, 201726]
                  ]

    for filename in filenames:

        # Load the model hyperparameters for file
        model_file_name = rename_file(filename, 'HyperParameters', 'pickle')
        os.chdir(result_path)
        model_params = pickle.load(open(model_file_name, 'rb'))

        # Read sku-sales data for forecasting
        os.chdir(data_path)
        df = pd.read_csv(filename)

        # 3100: FGB0723
        # 3100: FGB6542
        # 3100: FGB0737

        # df = df[df.ForecastUnitCode.isin(['3100:FGB0737'])]

        sku_group = df.groupby('ForecastUnitCode', as_index=False)
        sku_list = sku_group.groups.keys()

        total_predictions = []

        for sku in sku_list:
            df_sku = df[df.ForecastUnitCode.isin([sku])]
            period_index = 0
            print('-----------------------------------------------------')
            print('Result for SKU:', sku)
            for period in test_period:

                # data set to be fed for forecasting
                x_valid = df_sku[
                    (df_sku.ForecastWeek >= period[0]) &
                    (df_sku.ForecastWeek <= period[1])
                    ]
                x_valid_sales = x_valid['Weekly_Volume_Sales'].reset_index(drop=True)
                x_log = transform_data(x_valid_sales)
                history = [x for x in x_log]

                for model_param in model_params:
                    if model_param['sku'] == sku:
                        p_order, d_order, q_order = model_param['best_cfg']

                        if d_order > 0:
                            print('Difference SKU %s with order %d' % (sku, d_order))
                            # No second order differencing exists in our model, hence only 1st order is required
                            history = difference(history)

                        print('Period:', period_index + 1)
                        params = model_param['params']
                        residuals = model_param['residuals']
                        p = model_param['p']
                        q = model_param['q']
                        k_exog = model_param['k_exog']
                        k_trend = model_param['k_trend']
                        # intercept = model_param['intercept']
                        best_ols_params = model_param['best_ols_params']

                        steps = 4

                        y_actual = df_sku[
                            (df_sku.ForecastWeek >= oot_period[period_index][0]) & (
                                    df_sku.ForecastWeek <= oot_period[period_index][1])].reset_index(
                            drop=True)

                        if len(y_actual) > 4:
                            steps = 5

                        y_actual_sales = y_actual['Weekly_Volume_Sales']
                        # y_actual_log = np.log(y_actual_sales)

                        y_predicted_log = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog,
                                                                      endog=history, exog=None, start=len(history))

                        y_pred_promo_log = y_predicted_log.copy()

                        try:
                            if len(best_ols_params['pvalues'].keys()) > 1:
                                # SKU has promo significance
                                promo_sig_values = []
                                ols_pvalues_dict = best_ols_params['pvalues']
                                for index, row in y_actual.iterrows():
                                    # result = None
                                    # result = ols_pvalues_dict['const']
                                    result = 0
                                    for key, value in ols_pvalues_dict.items():

                                        if key != 'const':
                                            result += (row[key] * value)
                                    promo_sig_values.append(result)

                                y_pred_promo_log += promo_sig_values

                        except KeyError:
                            print('Key error %s' % sku)
                            continue



                        print()
                        print('-----------------')
                        print()

                        y_predicted = revert_to_order(y_predicted_log, x_log, d_order)
                        y_pred_series = pd.Series(y_predicted)

                        y_actual.drop(y_actual.columns[[0]], axis=1, inplace=True)
                        y_actual['Predicted_Weekly_Volume_Sales'] = y_pred_series
                        # print(y_real)
                        # print()

                        y_predicted_promo = revert_to_order(y_pred_promo_log, x_log, d_order)
                        y_pred_promo_series = pd.Series(y_predicted_promo)
                        y_actual['Promo_Weekly_Volume_Sales'] = y_pred_promo_series
                        total_predictions.append(y_actual)

                period_index += 1

        # Save predicted sales for respective files
        result_file_name = rename_file(filename, 'Result', 'csv')

        result_df = pd.concat(total_predictions)
        result_df.reset_index(drop=True, inplace=True)

        #Change path to result folder
        os.chdir(result_path)
        result_df.to_csv(result_file_name, sep=',')
        print('Forecasting completed for %s' %filename)


def train_with_residue(file_names=['Tea_Data_Set_1.csv']):
    """
    Trains ARIMA model post least MSE per sku & selects the best model
    Calculates Residue, performs Linear regression and then saves the model
    :param file_names: Training file, with default file name
    :return: None
    """

    for filename in file_names:
        # Change path to data folder to read file
        os.chdir(data_path)
        df = pd.read_csv(filename)

        # logic test
        # df = df[df.ForecastUnitCode.isin(['3100:FGB0737'])]
        #
        sku_group = df.groupby('ForecastUnitCode', as_index=False)
        sku_list = sku_group.groups.keys()

        sku_best_model = []

        for sku in sku_list:
            print()
            print(sku)

            # Select SKU to train & validate model
            df_train = df[df.ForecastUnitCode.isin([sku])]

            period_index = 0
            best_period_models = []

            for tp in train_period:
                print()
                print('Begin:%d End:%d' % (tp[0], tp[1]))
                print()
                # Select SKU data for training period
                df_train_period = df_train[
                    (df_train.ForecastWeek >= tp[begin]) & (df_train.ForecastWeek <= tp[end])
                    ]

                # Select SKU data from beginning to end of validation period
                df_validation_period = df_train[
                    (df_train.ForecastWeek >= validation_period[period_index][begin]) & (
                            df_train.ForecastWeek <= validation_period[period_index][end])
                    ]

                print('%d train samples for %d period.' % (len(df_train_period), (period_index + 1)))
                print('%d validation samples for %d period.' % (len(df_validation_period), (period_index + 1)))

                # Select sales data for training & validation
                train_sales = df_train_period['Weekly_Volume_Sales'].reset_index(drop=True)
                validation_sales = df_validation_period['Weekly_Volume_Sales'].reset_index(drop=True)
                train_valid_set = (train_sales, validation_sales)

                # Evaluate best model of selected train period
                best_score, best_cfg, best_params, best_residuals, best_p, best_q, best_k_exog, best_k_trend, best_intercept, y_predict_log = evaluate_models(
                    train_valid_set, p_range, d_range, q_range)

                # Perform OLS on resdiuals and promotion variables
                y_actual_log = transform_data(train_sales).as_matrix()
                y_predicted_log = y_predict_log[:len(train_sales)]

                y_residual_log = y_actual_log - y_predicted_log

                # print('Y_Actual:', y_actual)
                # print('Y_Actual_Log', y_actual_log)
                # print('Y_Predicted_Log', y_predicted_log)
                # print('Y_Residual_Log', y_residual_log)

                promo_cols = ['PM_Pipefill_Plinth', 'PM_Special_Pack_Extra_Free', 'PM_Multi_Buy',
                              'PM_Price_Reduction',
                              'PM_Special_Pack_Other', 'PM_Other_Mechanics', 'PM_Price_Reduction_Multi_Buy',
                              'PM_Special_Pack_Price_Marked_Pack']

                # Perform OLS
                endog = pd.DataFrame({'Residuals': copy.copy(y_residual_log)})
                exog = df_train_period[promo_cols].reset_index(drop=True)

                # Adding intercept
                # Adds a column to every row with the value: 1
                exog = sm.add_constant(exog)

                # Fit and summarize model
                ols_model = sm.OLS(endog=endog, exog=exog)
                results = ols_model.fit()
                # print(results.summary())
                _pvalues = results.pvalues
                # _pvalues.reset_index(inplace=True)
                # _temp = _pvalues[0]
                # _keys = results.pvalues.index.get_values()
                _pvalues_dict = results.pvalues.to_dict()

                # store intermediate values based on p significance
                promo_significant_dict = {}
                for key, value in _pvalues_dict.items():
                    # Ignore constant's pvalue
                    if key == 'const':
                        promo_significant_dict[key] = value
                    else:
                        if value < 0.05 and value > 0:
                            promo_significant_dict[key] = value

                r2_score = results.rsquared

                # model parameters to save
                best_period_model = {'best_cfg': best_cfg, 'mse': best_score, 'sku': sku, 'week': (period_index + 1),
                                     'residuals': best_residuals, 'p': best_p, 'q': best_q, 'k_exog': best_k_exog,
                                     'k_trend': best_k_trend, 'params': best_params, 'intercept': best_intercept,
                                     'promo_sig': promo_significant_dict, 'r2_score':r2_score }

                best_period_models.append(best_period_model)
                period_index += 1

            # Select best model in entire period
            best_model = find_best_model(best_period_models)

            # Add to best models list
            sku_best_model.append(best_model)
            print('____________________________________________________________________________________________')
            print('____________________________________________________________________________________________')

        # Save model to disk
        os.chdir(result_path)
        file = filename.split('.')
        new_file_name = file[0] + '_HyperParameters.pickle'
        save_model_to_disk(new_file_name, sku_best_model)

    print('Training completed')


# Config
data_path = 'C:\\Users\\prasenjit.giri\\OneDrive - Accenture\\Projects\\Python\\Unilever\\data'
result_path = 'C:\\Users\\prasenjit.giri\\OneDrive - Accenture\\Projects\\Python\\Unilever\\result'
original_file_name = 'Unilever_Tea_data_extract_till_wk_201730_ver2.csv'

# train_period = [[201420, 201648],
#                 [201424, 201652],
#                 [201428, 201704],
#                 [201432, 201708]
#                 ]

# New train period
train_period = [
    [201422, 201621],
    [201427, 201626],
    [201431, 201630],
    [201435, 201634],
    [201440, 201639]
]


# validation_period = [[201649, 201652],
#                      [201701, 201704],
#                      [201705, 201708],
#                      [201709, 201713]
#                      ]

# New in-time validation period
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

sales_files = ['Tea_Data_Set_1.csv', 'Tea_Data_Set_2.csv', 'Tea_Data_Set_3.csv']
promo_files = ['Tea_Data_Set_Promo_1.csv', 'Tea_Data_Set_Promo_2.csv', 'Tea_Data_Set_Promo_3.csv']

# Period indices
begin = 0
end = 1


def main():
    # Split & save data into smaller files
    # split_data_into_smaller_files(original_file_name)

    # Train models on all SKU
    # train()

    # Forecast on all SKU
    # forecast_oot()

    # Filter sku which has actual sales data (check if all period for sku has sales)
    # filter_record()

    # Find SKU which require differencing  (check for d>0)
    # find_sku_for_difference()

    # Calculate delta
    # calc_delta()

    # Generate results for the selected SKU
    # generate_results_for(filename='SKUs_Ops.xlsx')

    # Calculate delta for
    # calc_delta(filenames=['Result_SKUs_Ops.csv'])

    # Calculate4 delta for all files
    # result_file_names = ['Tea_Data_Set_1_Result.csv', 'Tea_Data_Set_2_Result.csv', 'Tea_Data_Set_3_Result.csv']
    # calc_delta(result_file_names)



    ## ARIMAX
    # Promo data file
    # prepare_data(original_file_name)

    # Residuals
    # OLS
    # Experiment
    # train_with_residue(promo_files)
    # predict_with_residues(promo_files)

    ##
    # filter_by_sku()

    # generate_data(original_file_name)

    deseasonalize_sales('Tea_Data_Set_Full.csv')


if __name__ == '__main__':
    main()
