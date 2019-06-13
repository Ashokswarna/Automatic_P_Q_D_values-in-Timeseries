"""
Created on 19 Mar 2018
Project: forecastlibrary
File: forecast
Author: prasenjit.giri
Copyright: Accenture AI
"""

import pandas as pd
import numpy as np
import copy
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
from utility import get_window_periods, get_date_from_year_week
from data_helpers import load_data, load_model


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


def transform_data(data):
    data_log = np.log(data)
    data_log[data_log == -np.inf] = 0
    data_log[data_log == np.inf] = 0
    return data_log


def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def _monthly_sku_forecast(df, category, country):
    sku_list = df.groupby('sku', as_index=False).groups.keys()
    monthly_results = []

    for sku in sku_list:
        df_sku = df[df['sku'].isin([sku])]
        # df_sku['forecastWeek'] = pd.to_datetime(df_sku['forecastWeek'].astype(str) + '1', format='%Y%W%w')
        df_sku['forecastWeek'] = df_sku.apply(lambda x: get_date_from_year_week(x['forecastWeek']), axis=1)

        df_sku = df_sku.groupby(['sku', df_sku['forecastWeek'].dt.to_period('m')]).sum().reset_index()
        df_sku['category'] = category
        df_sku['market'] = country
        df_sku.rename(columns={'forecastWeek': 'month'}, inplace=True)
        df_sku['month'] = df_sku['month'].apply(lambda x: str(x))
        monthly_results.append(df_sku)

    res_df = pd.concat(monthly_results)
    return res_df


def _monthly_ret_sku_forecast(df, category, country):
    sku_list = df.groupby('sku', as_index=False).groups.keys()
    monthly_results = []

    for sku in sku_list:
        df_sku = df[df['sku'].isin([sku])]
        # df_sku['forecastWeek'] = pd.to_datetime(df_sku['forecastWeek'].astype(str) + '1', format='%Y%W%w')
        df_sku['forecastWeek'] = df_sku.apply(lambda x: get_date_from_year_week(x['forecastWeek']), axis=1)

        df_sku = df_sku.groupby(
            ['sku', 'accountPlanningGroupCode', df_sku['forecastWeek'].dt.to_period('m')]).sum().reset_index()
        df_sku['category'] = category
        df_sku['market'] = country
        df_sku.rename(columns={'forecastWeek': 'month'}, inplace=True)
        df_sku['month'] = df_sku['month'].apply(lambda x: str(x))
        monthly_results.append(df_sku)

    res_df = pd.concat(monthly_results)

    return res_df


def _forecast_sku(df, models, analysis_windows, category, country):
    # Future forecast at week level
    sku_list = df.groupby('sku', as_index=False).groups.keys()

    obp = analysis_windows['future_forecast']['obs_window_future']
    fp = analysis_windows['future_forecast']['future_window']
    f_steps = analysis_windows['future_forecast']['forecast_steps']
    f_select = analysis_windows['future_forecast']['forecast_select']

    total_predictions = []

    for sku in sku_list:
        df_sku = df[df['sku'].isin([sku])]

        x_train = df_sku[(df_sku['forecastWeek'] >= obp[0]) & (df_sku['forecastWeek'] <= obp[1])]
        x_train = x_train['actualVolume'].reset_index(drop=True)
        x_log = transform_data(x_train)
        obs_mat = [x for x in x_log]

        for model in models:
            if model['sku'] == sku:
                p_order, d_order, q_order = model['best_cfg']

                if d_order > 0:
                    print('Difference SKU %s with order %d' % (sku, d_order))
                    obs_mat = difference(obs_mat)

                params = model['params']
                residuals = model['residuals']
                p = model['p']
                q = model['q']
                k_trend = model['k_trend']
                k_exog = model['k_exog']

                # Forecast
                y_pred_log = _arma_predict_out_of_sample(params=params, steps=f_steps, errors=residuals,
                                                         p=p, q=q, k_trend=k_trend, k_exog=k_exog, endog=obs_mat,
                                                         start=len(obs_mat)
                                                         )
                y_pred = revert_to_order(y_pred_log, x_log, d_order)
                y_pred_series = pd.Series(y_pred)

                # Select the data to which we will append the forecast volumes
                y_hat = df_sku[
                    (df_sku['forecastWeek'] >= fp[0]) & (df_sku['forecastWeek'] <= fp[1])].reset_index(drop=True)
                y_hat['forecastVolume'] = round(y_pred_series, 0)

                total_predictions.append(y_hat)

    res_week_forecast = pd.concat(total_predictions)
    res_week_forecast.reset_index(drop=True, inplace=True)

    res_month_forecast = _monthly_sku_forecast(df=res_week_forecast, category=category, country=country)

    res = {
        'weeklyForecast': res_week_forecast,
        'monthlyForecast': res_month_forecast
    }

    return res


def _forecast_retailer_sku(dfs, dfr, analysis_windows, category, country):
    # select sku, sales, week for forecast sales
    cols = ['sku', 'forecastWeek', 'actualVolume']

    # select observation period sales data
    obp = analysis_windows['future_forecast']['obs_window_future']
    dfr_f = dfr[(dfr['forecastWeek'] >= obp[0]) & (dfr['forecastWeek'] <= obp[1])]

    # Group to sales level of observation window
    df_wk_sales = dfr_f.groupby(['sku', 'forecastWeek'], as_index=False)['actualVolume'].sum()

    # Find ratio of retailer sales from observation window
    df_merge = pd.merge(df_wk_sales, dfr_f, on=['sku', 'forecastWeek'], how='left')

    # Ratio of sales distribution per retailer at week level
    df_merge['ratio'] = df_merge.apply(
        lambda row: (round((row.actualVolume_y / row.actualVolume_x) * 100)) if (row.actualVolume_y > 0) else 0, axis=1)

    # Ratio of sales distribution per retailer across observation window week
    df_ret_avg_ratio = df_merge.groupby(['sku', 'accountPlanningGroupCode'], as_index=False)['ratio'].mean().round()

    df_dist = pd.merge(dfs, df_ret_avg_ratio, on=['sku'], how='left')
    df_dist['retForecastVolume'] = df_dist.apply(lambda row: ((row.forecastVolume * row.ratio) / 100), axis=1).round()

    df_res_sku_dist = _monthly_ret_sku_forecast(df_dist, category=category, country=country)

    res = {
        'weeklyForecast': df_dist.drop_duplicates(),
        'monthlyForecast': df_res_sku_dist.drop_duplicates()
    }

    return res


def _forecast_sku_analysis(df, models, analysis_windows, category, country):

    a_period = analysis_windows['hist_analysis']
    sku_list = df.groupby('sku', as_index=False).groups.keys()

    total_predictions = []

    for sku in sku_list:
        df_sku = df[df['sku'].isin([sku])]

        for model in models:
            if model['sku'] == sku:

                for ap in a_period:
                    obp = ap['analysis_window']
                    # fp = analysis_windows['future_forecast']['future_window']
                    f_steps = ap['pred_steps']
                    f_select = ap['analysis_period']

                    p_order, d_order, q_order = model['best_cfg']

                    x_train = df_sku[(df_sku['forecastWeek'] >= obp[0]) & (df_sku['forecastWeek'] <= obp[1])]
                    x_train = x_train['actualVolume'].reset_index(drop=True)
                    x_log = transform_data(x_train)
                    obs_mat = [x for x in x_log]

                    if d_order > 0:
                        print('Difference SKU %s with order %d' % (sku, d_order))
                        obs_mat = difference(obs_mat)

                    params = model['params']
                    residuals = model['residuals']
                    p = model['p']
                    q = model['q']
                    k_trend = model['k_trend']
                    k_exog = model['k_exog']

                    # Forecast
                    y_pred_log = _arma_predict_out_of_sample(params=params, steps=f_steps, errors=residuals,
                                                             p=p, q=q, k_trend=k_trend, k_exog=k_exog, endog=obs_mat,
                                                             start=len(obs_mat)
                                                             )
                    y_pred = revert_to_order(y_pred_log, x_log, d_order)
                    y_pred = y_pred[-1:]
                    y_pred_series = pd.Series(y_pred)

                    # Select the data to which we will append the forecast volumes
                    y_hat = df_sku[
                        (df_sku['forecastWeek'] == f_select)].reset_index(drop=True)
                    # y_hat = df_sku[
                    #     (df_sku['forecastWeek'] >= fp[0]) & (df_sku['forecastWeek'] <= fp[1])].reset_index(drop=True)
                    y_hat['forecastVolume'] = round(y_pred_series, 0)

                    total_predictions.append(y_hat)

    res_week_forecast = pd.concat(total_predictions)
    res_week_forecast.reset_index(drop=True, inplace=True)

    # res_month_forecast = _monthly_sku_forecast(df=res_week_forecast, category=category, country=country)

    res = {
        'weeklyForecast': res_week_forecast
        # 'monthlyForecast': res_month_forecast
    }

    return res


def calc_error(actual, forecast):
    error = actual - forecast
    return error


def calc_ape(actual, forecast):
    ape = 0
    if actual > 0:
        ape = abs(round(((actual - forecast) / actual) * 100, 0))
        if ape > 100:
            ape = 99
        elif ape < 0:
            ape = 0
    return ape


def calc_bias(actual, forecast):
    bias = 0
    if forecast > 0:
        # bias = round((((actual - forecast)/forecast) * 100), 0)
        bias = round((((forecast - actual) / forecast) * 100), 0)
        if bias > 100:
            bias = 99
        elif bias < -100:
            bias = -99
    return bias


def calc_accuracy(actual, forecast):
    accuracy = 0
    if actual > 0:
        accuracy = abs(round(((1 - ((actual - forecast) / actual)) * 100), 0))
        if accuracy > 100:
            accuracy = 100
        elif accuracy < 0:
            accuracy = 0
    return accuracy


####################################### Retailer SKU Analysis Section ###############################################

def ret_sku_bias_calc(actual, forecast):

    if forecast == 34:
        print('FOUND!')

    adj_actual = actual
    if actual == 0:
        adj_actual = 1

    adj_forecast = forecast
    if forecast == 0:
        adj_forecast = 1

    bias = round( ((adj_forecast - adj_actual)/adj_forecast) * 100, 0)
    adj_bias = bias

    if bias < -100:
        adj_bias = -100
    elif bias > 100:
        adj_bias = 100

    return adj_bias


def ret_sku_ape_calc(actual, forecast):
    adj_actual = actual
    if actual == 0:
        adj_actual = 1

    adj_forecast = forecast
    if forecast == 0:
        adj_forecast = 1

    ape = round(((adj_actual -adj_forecast)/adj_actual) * 100, 0)
    ape = abs(ape)

    adj_ape = ape
    if ape > 100:
        adj_ape = 100

    return adj_ape







####################################### Retailer SKU Analysis Section ###############################################

def _forecast_retailer_sku_analysis(dfs, dfr, analysis_windows, category, country, future):

    _future_analysis = future[['sku', 'forecastWeek', 'category', 'market', 'retForecastVolume', 'accountPlanningGroupCode']].copy()
    _future_analysis.rename(columns={'retForecastVolume': 'forecastVolume'}, inplace=True)
    _future_analysis['actualVolume'] = np.nan
    _future_analysis['bias'] = np.nan
    _future_analysis['extremeBias'] = np.nan
    _future_analysis['ape'] = np.nan
    _future_analysis['forecastError'] = np.nan
    _future_analysis['accuracy'] = np.nan
    _future_analysis.fillna('N/A', inplace=True)

    a_period = analysis_windows['hist_analysis']
    sku_list = dfs.groupby('sku', as_index=False).groups.keys()
    cols = ['sku', 'forecastWeek', 'category', 'market', 'forecastVolume', ]
    total_result = []

    for sku in sku_list:
        df_sku = dfs[dfs['sku'].isin([sku])]
        # Select only relevant columns
        df_sku = df_sku[cols]
        dfr_sku = dfr[dfr['sku'].isin([sku])]
        for week in df_sku['forecastWeek']:
            df_wk_sku = df_sku[df_sku['forecastWeek'].isin([week])]

            # get actuals for retailer for current week
            # select sales for current week
            dfr_wk_ret = dfr_sku[dfr_sku['forecastWeek'].isin([week])]
            # select only relevant columns
            dfr_wk_ret = dfr_wk_ret[['sku', 'accountPlanningGroupCode', 'actualVolume']]

            # Join the actual with forecast table
            df_sku_ret_wk_sales = pd.merge(df_wk_sku, dfr_wk_ret, on='sku')

            for ap in a_period:
                if ap['analysis_period'] == week:
                    # Get the relevant observation window
                    obp = ap['analysis_window']
                    dfr_f = dfr_sku[(dfr_sku['forecastWeek'] >= obp[0]) & (dfr['forecastWeek'] <= obp[1])]

                    # Group to sales level of observation window
                    df_wk_sales = dfr_f.groupby(['sku', 'forecastWeek'], as_index=False)['actualVolume'].sum()

                    # Find ratio of retailer sales from observation window
                    df_merge = pd.merge(df_wk_sales, dfr_f, on=['sku', 'forecastWeek'], how='left')

                    # Ratio of sales distribution per retailer at week level
                    df_merge['ratio'] = df_merge.apply(
                        lambda row: (round((row.actualVolume_y / row.actualVolume_x) * 100)) if (
                                row.actualVolume_y > 0) else 0, axis=1)

                    # Ratio of sales distribution per retailer across observation window week
                    df_ret_avg_ratio = df_merge.groupby(['sku', 'accountPlanningGroupCode'], as_index=False)[
                        'ratio'].mean().round()

                    df_dist = pd.merge(df_sku_ret_wk_sales, df_ret_avg_ratio, on=['sku', 'accountPlanningGroupCode'], how='left')
                    df_dist['actualVolume'] = round(df_dist['actualVolume'], 0)

                    df_dist['retForecastVolume'] = round(df_dist.apply(
                        lambda row: ((row.forecastVolume * row.ratio) / 100) if (row.ratio > 0) else 0, axis=1), 0)

                    disp_cols = ['sku', 'forecastWeek', 'category', 'accountPlanningGroupCode', 'market', 'actualVolume',
                            'retForecastVolume', ]

                    df_dist = df_dist[disp_cols]
                    df_dist.rename(columns={'retForecastVolume': 'forecastVolume'}, inplace=True)

                    df_dist['forecastError'] = df_dist.apply(lambda x: calc_error(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)
                    # df_dist['ape'] = df_dist.apply(lambda x: calc_ape(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)
                    # df_dist['bias'] = df_dist.apply(lambda x: calc_bias(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)

                    df_dist['ape'] = df_dist.apply(
                        lambda x: ret_sku_ape_calc(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)
                    df_dist['bias'] = df_dist.apply(
                        lambda x: ret_sku_bias_calc(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)

                    df_dist['extremeBias'] = df_dist.apply(lambda x: calc_extreme_bias(x['actualVolume'], x['forecastVolume']), axis=1)
                    df_dist['accuracy'] = round((100 - df_dist['ape']), 0)

                    total_result.append(df_dist)


    # REGION  Future retailer-sales analysis
    f_period = analysis_windows['future_forecast']

    res = pd.concat(total_result)
    res.drop_duplicates(inplace=True)

    res.fillna(0, inplace=True)
    res.replace([np.inf, -np.inf], 0, inplace=True)

    res_disp = res.copy()
    res_disp['accuracy'] = res_disp.apply(lambda x: (str(int(x['accuracy'])) + '%'), axis=1)
    res_disp['bias'] = res_disp.apply(lambda x: (str(int(x['bias'])) + '%'), axis=1)
    res_disp['ape'] = res_disp.apply(lambda x: (str(int(x['ape'])) + '%'), axis=1)



    # append future & past forecast
    ret_sku_table_df = pd.concat([_future_analysis.sort_values(by='forecastWeek', ascending=False), res_disp])
    ret_sku_table_df = ret_sku_table_df.drop_duplicates()

    res_acc_graph = round(res.groupby('accountPlanningGroupCode', as_index=False)['accuracy'].mean(), 0)
    res_bias_graph = round(res.groupby('accountPlanningGroupCode', as_index=False)['bias'].mean(),0)

    pos_y = []
    pos_x = []
    # pos_base = []

    neg_y = []
    neg_x = []
    neg_base = []

    for index, row in res_bias_graph.iterrows():
        if row['bias'] < 0:
            neg_y.append(row['accountPlanningGroupCode'])
            neg_x.append(- row['bias'])
            neg_base.append(row['bias'])
        elif row['bias'] > 0:
            pos_y.append(row['accountPlanningGroupCode'])
            pos_x.append(row['bias'])
        else:
            continue

        # if row['bias'] >= 0:
        #     pos_y.append(row['accountPlanningGroupCode'])
        #     pos_x.append(row['bias'])
        #     # pos_base.append(row['bias'])
        # else:
        #     neg_y.append(row['accountPlanningGroupCode'])
        #     neg_x.append(- row['bias'])
        #     neg_base.append(row['bias'])

    pos_graph = {
        'y': pos_y,
        'x': pos_x,
        # 'base': pos_base
        'base': 0
    }

    neg_graph = {
        'y': neg_y,
        'x': neg_x,
        'base': neg_base
    }

    bias_graph = {
        'bias_graph_pos': pos_graph,
        'bias_graph_neg': neg_graph
    }

    acc_x = []
    acc_y = []

    for index, row in res_acc_graph.iterrows():
        if row['accuracy'] > 0:
            acc_x.append(row['accountPlanningGroupCode'])
            acc_y.append(round(row['accuracy'], 0))


    # accuracy_graph = {
    #     'x': np.array(res_acc_graph['accountPlanningGroupCode']).tolist(),
    #     'y': np.array(round(res_acc_graph['accuracy'], 0)).tolist()
    # }

    accuracy_graph = {
        'x': acc_x,
        'y': acc_y
    }

    result = {
        # 'table': res,
        # 'table': res_disp,
        'table': ret_sku_table_df,
        'accuracy_graph': accuracy_graph,
        'bias_graph': bias_graph
    }

    return result


def forecast_oot(req_dict):
    # Get all future forecast , future forecast analysis, retailer forecast, retailer forecast analysis
    future_steps = int(req_dict.futurePeriod)
    past_steps = int(req_dict.historicalPeriod)
    start = req_dict.forecastStart
    analysis_windows = get_window_periods(start, future_steps=future_steps, past_steps=past_steps)

    # Load data & model
    filename = req_dict.data
    country = req_dict.market
    category = req_dict.category
    skus = req_dict.sku

    df_skus, df_retailers = load_data(filename=filename, skus=skus, country=country, category=category)

    models = load_model()

    forecast_sku = _forecast_sku(df=df_skus, models=models, analysis_windows=analysis_windows,
                                 category=category, country=country)

    disp_sku_for_cols = ['sku', 'market', 'category', 'forecastWeek', 'forecastVolume']
    sku_forecast_week = forecast_sku['weeklyForecast']
    sku_list = (sku_forecast_week['sku'].unique()).tolist()

    sku_forecast_week = sku_forecast_week[disp_sku_for_cols].to_dict('records')

    disp_sku_for_cols = ['sku', 'market', 'category', 'month', 'forecastVolume']
    sku_forecast_month = forecast_sku['monthlyForecast']
    sku_forecast_month = sku_forecast_month[disp_sku_for_cols].to_dict('records')

    forecast_retailer_sku = _forecast_retailer_sku(dfs=forecast_sku['weeklyForecast'], dfr=df_retailers,
                                                   analysis_windows=analysis_windows, category=category,
                                                   country=country)

    disp_ret_sku_for_cols = ['sku', 'forecastWeek', 'category', 'market', 'accountPlanningGroupCode',
                             'retForecastVolume']
    ret_sku_forecast_week = forecast_retailer_sku['weeklyForecast']

    ret_list = (ret_sku_forecast_week['accountPlanningGroupCode'].unique()).tolist()
    ret_sku_analysis = forecast_retailer_sku['weeklyForecast'].copy()

    ret_sku_forecast_week = ret_sku_forecast_week[disp_ret_sku_for_cols]
    ret_sku_forecast_week.rename(columns={'retForecastVolume': 'forecastVolume'}, inplace=True)
    ret_sku_forecast_week = ret_sku_forecast_week.to_dict('records')

    ret_sku_forecast_month = forecast_retailer_sku['monthlyForecast']
    disp_ret_sku_for_cols = ['sku', 'month', 'category', 'market', 'accountPlanningGroupCode',
                             'retForecastVolume']
    ret_sku_forecast_month = ret_sku_forecast_month[disp_ret_sku_for_cols]
    ret_sku_forecast_month.rename(columns={'retForecastVolume': 'forecastVolume'}, inplace=True)
    ret_sku_forecast_month = ret_sku_forecast_month.to_dict('records')

    forecast_sku_analysis = _forecast_sku_analysis(df=df_skus, models=models, analysis_windows=analysis_windows,
                                                   category=category, country=country)

    _ft_sku_act = forecast_sku_analysis['weeklyForecast'].copy()

    _ft_sku_act['actualVolume'] = round(_ft_sku_act['actualVolume'], 0)
    _ft_sku_act['forecastVolume'] = round(_ft_sku_act['forecastVolume'], 0)

    _ft_sku_act['forecastError'] = _ft_sku_act.apply(lambda x: calc_error(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)
    _ft_sku_act['ape'] = _ft_sku_act.apply(lambda x: calc_ape(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)
    _ft_sku_act['accuracy'] = round((100 - _ft_sku_act['ape']), 0)
    _ft_sku_act['bias'] = _ft_sku_act.apply(lambda x: calc_bias(actual=x['actualVolume'], forecast=x['forecastVolume']), axis=1)
    _ft_sku_act['extremeBias'] = _ft_sku_act.apply(lambda x: calc_extreme_bias(x['actualVolume'], x['forecastVolume']),
                                                   axis=1)

    overall_accuracy = round(_ft_sku_act['accuracy'].mean(), 0)
    # overall_bias = round(_ft_sku_act['bias'].mean(), 0)
    overall_bias = calc_total_bias(_ft_sku_act)

    __sku_forecast = forecast_sku['weeklyForecast']
    __sku_forecast['actualVolume'] = np.nan

    # sku_analysis_df = pd.concat([_ft_sku_act, __sku_forecast])
    # Append % to ape, bias, accuracy
    _ft_sku_act_disp = _ft_sku_act.copy()
    _ft_sku_act_disp['accuracy'] = _ft_sku_act_disp.apply(lambda x: (str(int(x['accuracy'])) + '%'), axis=1)
    _ft_sku_act_disp['bias'] = _ft_sku_act_disp.apply(lambda x: (str(int(x['bias'])) + '%'), axis=1)
    _ft_sku_act_disp['ape'] = _ft_sku_act_disp.apply(lambda x: (str(int(x['ape'])) + '%'), axis=1)

    # sku_analysis_df = pd.concat([__sku_forecast.sort_values(by='forecastWeek', ascending=False), _ft_sku_act])
    sku_analysis_df = pd.concat([__sku_forecast.sort_values(by='forecastWeek', ascending=False), _ft_sku_act_disp])
    sku_analysis_df.fillna('N/A', inplace=True)
    sku_analysis_df = sku_analysis_df.drop_duplicates()

    sku_analysis_df_dict = sku_analysis_df.to_dict('records')

    # # Total Graph
    act_graph_sales = _ft_sku_act.groupby('forecastWeek', as_index=False)['actualVolume'].sum()
    act_graph_week = act_graph_sales['forecastWeek'].apply(lambda x: format_graph_week(x))
    act_graph_sales = act_graph_sales['actualVolume']

    # Split forecast in actual-forecast & future-forecast
    df_act_fr = sku_analysis_df[sku_analysis_df['forecastWeek'] < int(start)].copy()
    df_act_fr = df_act_fr.groupby('forecastWeek', as_index=False)['forecastVolume'].sum()
    fct_graph_week = df_act_fr['forecastWeek'].apply(lambda x: format_graph_week(x))
    fct_graph_sales = df_act_fr['forecastVolume']

    df_fut_fr = sku_analysis_df[sku_analysis_df['forecastWeek'] >= int(start)].copy()
    df_fut_fr = df_fut_fr.groupby('forecastWeek', as_index=False)['forecastVolume'].sum()
    fut_fr_graph_week = df_fut_fr['forecastWeek'].apply(lambda x: format_graph_week(x))
    fut_fr_graph_sales = df_fut_fr['forecastVolume']

    # fct_graph_sales = sku_analysis_df.groupby('forecastWeek', as_index=False)['forecastVolume'].sum()
    # fct_graph_week = fct_graph_sales['forecastWeek']
    # fct_graph_sales = fct_graph_sales['forecastVolume']

    dict_graph = {}

    dict_graph['actualVolume'] = {
        'x': np.array(act_graph_week).tolist(),
        'y': np.array(act_graph_sales).tolist()
    }
    dict_graph['forecastVolume'] = {
        'x': np.array(fct_graph_week).tolist(),
        'y': np.array(fct_graph_sales).tolist()
    }
    dict_graph['futureForecastVolume'] = {
        'x': np.array(fut_fr_graph_week).tolist(),
        'y': np.array(fut_fr_graph_sales).tolist()
    }

    ## REGION: Forecast Retailer SKU Analysis
    res_ret_sku_analysis = _forecast_retailer_sku_analysis(dfs=forecast_sku_analysis['weeklyForecast'],
                                                           dfr=df_retailers, analysis_windows=analysis_windows,
                                                           category=category, country=country, future=ret_sku_analysis)

    # res_ret_sku_analysis_dict = res_ret_sku_analysis['table'].to_dict('records')

    res_ret_sku_analysis_dict = res_ret_sku_analysis['table']
    # res_ret_sku_analysis_dict.rename(columns={'retForecastVolume': 'forecastVolume'}, inplace=True)
    res_ret_sku_analysis_dict = res_ret_sku_analysis_dict.to_dict('records')

    result = {
        'forecastResults': {
            'forecastSku': {
                'weeklyForecast': sku_forecast_week,
                'monthlyForecast': sku_forecast_month
                # 'skuList': sku_list
            },
            'forecastRetailerSku': {
                'weeklyForecast': ret_sku_forecast_week,
                'monthlyForecast': ret_sku_forecast_month
                # 'skuList': sku_list,
                # 'retailerList': ret_list
            },
            'forecastSkuAnalysis': {
                'table': sku_analysis_df_dict,
                'graph': dict_graph,
                'totalAccuracy': overall_accuracy,
                'totalBias': overall_bias
                # 'skuList': sku_list
            },
            'forecastRetailerSkuAnalysis': {
                'table': res_ret_sku_analysis_dict,
                'graph': {
                    'forecastAccuracy': res_ret_sku_analysis['accuracy_graph'],
                    'forecastBias': res_ret_sku_analysis['bias_graph']
                }
                # 'skuList': sku_list,
                # 'retailerList': ret_list
            }
        }
    }
    return result


def format_graph_week(week_number):
    str_wk = str(week_number)
    result = str_wk[:4] + '-wk' + str_wk[-2:]
    return result


def calc_extreme_bias(actual, predicted):
    threshold = 0.3
    try:
        if ((actual / predicted) - 1) > threshold or ((actual / predicted) - 1) < -threshold:
            return actual
        else:
            return 0
    except ZeroDivisionError:
        return 0


def safe_div(x,y):
    if y == 0:
        return 0
    return x/y


def calc_total_bias(df):
    result = []

    sku_list = df.groupby('sku', as_index=False).groups.keys()

    for sku in sku_list:
        df_sku = df[df['sku'].isin([sku])]
        pos_bias = []
        neg_bias = []

        for index, row in df_sku.iterrows():
            if row['bias'] > 0:
                pos_bias.append(row['bias'])
            else:
                neg_bias.append(row['bias'])

        pos_bias_count = len(pos_bias)
        neg_bias_count = len(neg_bias)

        pos_bias_weight = safe_div(x=pos_bias_count, y=(pos_bias_count + neg_bias_count))
        neg_bias_weight = safe_div(x=neg_bias_count, y=(pos_bias_count + neg_bias_count))

        pos_avg_bias = safe_div(x=sum(pos_bias), y=pos_bias_count)
        neg_avg_bias = safe_div(x=sum(neg_bias), y=neg_bias_count)

        pos_wt_avg_bias = pos_bias_weight * pos_avg_bias
        neg_wt_avg_bias = neg_bias_weight * neg_avg_bias

        total_wt_avg_bias = pos_wt_avg_bias + neg_wt_avg_bias
        result.append(total_wt_avg_bias)

    wt_avg_bias = safe_div(x=sum(result), y=len(result))
    return round(wt_avg_bias, 0)


def main():
    forecast_oot('df_data_001.csv')


if __name__ == '__main__':
    main()
