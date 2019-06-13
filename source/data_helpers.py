"""
Created on 16 Mar 2018
Project: forecastlibrary
File: data_helpers
Author: prasenjit.giri
Copyright: Accenture AI
"""

import pandas as pd
import numpy as np
import pickle
from os import path
from app_settings import read_config


def load_data(filename, skus, country, category):
    app_settings = read_config()
    data_path = app_settings['data_path']
    data_file_path = path.join(data_path, filename)

    df = pd.read_csv(data_file_path)

    df.rename(columns={'Sku': 'sku', 'Sales': 'actualVolume', 'Week': 'forecastWeek',
                       'Retailer': 'accountPlanningGroupCode', 'Market': 'market',
                       'Category': 'category'}, inplace=True)

    cols = ['sku', 'actualVolume', 'forecastWeek', 'accountPlanningGroupCode', 'market', 'category']

    df = df[df['market'] == country]
    df = df[df['category'] == category]
    df = df[df['sku'].isin(skus)]

    df = df[cols]

    df_sku_sales = df.groupby(['sku', 'forecastWeek'], as_index=False)['actualVolume'].sum()
    df_sku_sales['category'] = category
    df_sku_sales['market'] = country

    return df_sku_sales, df


def load_model():
    app_settings = read_config()
    model_path = app_settings['model_path']
    model_file_name = app_settings['model_file']
    model_file_path = path.join(model_path, model_file_name)
    model = pickle.load(open(model_file_path, 'rb'))
    return model


def log_transform(data):
    data_log = np.log(data)
    data_log[data_log == -np.inf] = 0
    data_log[data_log == np.inf] = 0
    return data_log


def revert_log_transform(y_log, x_log, d_order=0):
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
