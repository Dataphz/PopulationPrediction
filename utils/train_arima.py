import datetime 
import time 
import warnings
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from tqdm import tqdm
import os

from utils.common_utils import checkdir
from utils.log import log 
from core.config import PreliminaryConfig as cfg 

def train_auto_arimas():
    train_auto_arima(cfg.FILE.front_flow_file, middle=True)
    train_auto_arima(cfg.FILE.final_flow_file, middle=False)

def train_auto_arima(flow_file, middle=True):

    date_list = test_date_generate(middle)
    if middle:
        days = 5
    else:
        days = 10

    flow_df = pd.read_csv(flow_file)
    flow_df = flow_df.sort_values(by=['city_code','district_code','date_dt'])
    district_code_list = flow_df['district_code'].unique()

    preds_df = pd.DataFrame()
    for district_code in  district_code_list:
        district_df = flow_df[flow_df['district_code'] == district_code]
        city_code = district_df['city_code'].iloc[0]
        predict_columns = ['dwell','flow_in','flow_out']
        tmp_df = pd.DataFrame(data=date_list, columns=['date_dt'])
        tmp_df['city_code'] = city_code
        tmp_df['district_code'] = district_code
        
        for column in predict_columns:
            ts_log = np.log(1 + district_df[column].values)
            arima_model = auto_arima(ts_log, start_p=1, max_p=9, start_q=1, max_q=9, max_d=7,
                                 start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=7,
                                 m=7, random_state=2018,
                                 trace=True,
                                 seasonal=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)
            # arima_model = auto_arima(ts_log, start_p=1, max_p=9, start_q=1, max_q=9, max_d=15,
            #                         start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=15,
            #                         m=15, random_state=2018,
            #                         trace=True,
            #                         seasonal=True,
            #                         error_action='ignore',
            #                         suppress_warnings=True,
            #                         stepwise=True)
            preds = arima_model.predict(n_periods=days)
            preds = pd.Series(preds)
            preds = np.exp(preds) - 1
            tmp_df = pd.concat([tmp_df, preds], axis=1)
        
        tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)
        
    trian_time = time.strftime("%m-%d-%H", time.localtime())
    result_dir = checkdir(os.path.join(cfg.LOG.home_log_dir, '{}_ARIMA'.format(trian_time)))

    if middle:
        preds_df.to_csv(os.path.join(result_dir, 'arima_middle_prediction.csv'), index=False, header = False)
    else:
        preds_df.to_csv(os.path.join(result_dir, 'arima_final_prediction.csv'), index=False, header = False)

def test_date_generate(middle = True):

    if middle:
        init_date = datetime.date(2017,8,19)
        days = 5
    else:
        init_date = datetime.date(2017,11,5)
        days = 10

    date_list = list()
    for delta in range(days):
        _date = init_date + datetime.timedelta(days = delta)
        date_list.append(_date.strftime('%Y%m%d'))
    return date_list