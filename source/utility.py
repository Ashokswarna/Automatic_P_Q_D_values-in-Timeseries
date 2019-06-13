"""
Created on 26 Apr 2018
Project: forecastlibrary
File: utility
Author: prasenjit.giri
Copyright: Accenture AI
"""


from datetime import datetime, timedelta


def convert_to_date(date_string, date_format='%Y%W%w'):
    _ds = date_string + '1'
    dt = datetime.strptime(_ds, date_format)
    return dt


def convert_date_to_int(date, date_format='%Y%W'):
    date_string = datetime.strftime(date, date_format)
    return int(date_string)


def observation_window(year_week):
    lag = 4
    input_period = 8
    dt = convert_to_date(year_week)
    obs_start = dt - timedelta(weeks=lag+input_period)
    obs_end = obs_start + timedelta(weeks=input_period-1)
    result = [convert_date_to_int(obs_start), convert_date_to_int(obs_end)]
    return result


def history_window(year_week, past_steps):
    dt = convert_to_date(year_week)

    # Assumptions: User is always 4 weeks behind exclusive of the selected date
    lag = 4
    dt = dt - timedelta(weeks=lag)

    analysis_periods = []
    for i in range(0, past_steps):
        end_date = dt - timedelta(weeks=i+1)
        end_date_int = convert_date_to_int(end_date)
        obs = observation_window(year_week=str(end_date_int))
        pred_steps = int((end_date - convert_to_date(str(obs[1]))).days / 7)
        analysis_period = {
            'analysis_period': end_date_int,
            'analysis_window': obs,
            'pred_steps': pred_steps
        }
        analysis_periods.append(analysis_period)

    return analysis_periods


def future_window(year_week, steps):
    dt = convert_to_date(year_week)
    future_date = convert_date_to_int(dt + timedelta(weeks=steps - 1))
    result = [int(year_week), future_date]
    return result


def get_window_periods(year_week, future_steps, past_steps):

    # Calculate the forecast period
    f_window = future_window(year_week, future_steps)
    obs_window_future = observation_window(year_week)
    forecast_steps = 4 + future_steps

    future_forecast_dict = {
        'future_window': f_window,
        'obs_window_future': obs_window_future,
        'forecast_steps': forecast_steps,
        'forecast_select': future_steps
    }

    # Calculate the historical period
    h_window = history_window(year_week=year_week, past_steps=past_steps)

    result = {
        'future_forecast': future_forecast_dict,
        'hist_analysis': h_window
    }

    return result


def get_date_from_year_week(year_week):
    weeks_month_map = [
        {
            'weeks': [1, 2, 3, 4],
            'month': 1
        },
        {
            'weeks': [5, 6, 7, 8],
            'month': 2
        },
        {
            'weeks': [9, 10, 11, 12, 13],
            'month': 3
        },
        {
            'weeks': [14, 15, 16, 17],
            'month': 4
        },
        {
            'weeks': [18, 19, 20, 21],
            'month': 5
        },
        {
            'weeks': [22, 23, 24, 25, 26],
            'month': 6
        },
        {
            'weeks': [27, 28, 29, 30],
            'month': 7
        },
        {
            'weeks': [31, 32, 33, 34],
            'month': 8
        },
        {
            'weeks': [35, 36, 37, 38, 39],
            'month': 9
        },
        {
            'weeks': [40, 41, 42, 43],
            'month': 10
        },
        {
            'weeks': [44, 45, 46, 47],
            'month': 11
        },
        {
            'weeks': [48, 49, 50, 51, 52],
            'month': 12
        }
    ]

    month = None
    _year_week = str(year_week)
    d_year = _year_week[:4]
    wk_year = _year_week[-2:]
    wk_year_int = int(wk_year)

    for wmm in weeks_month_map:
        for week in wmm['weeks']:
            if wk_year_int == week:
                month = wmm['month']
                break
    date_str = d_year + str(month) + '1'

    dt = datetime.strptime(date_str, '%Y%m%d')
    return dt
