import os
import pandas as pd 
import numpy as np 

# from core.config import PreliminaryConfig as cfg 
# from dataset.data_utils import DataGenerator

def city_preds_merge():
    p = 199
    final_flow_df = pd.read_csv('../../data/final_flow_train.csv')
    city_code_list = final_flow_df['city_code'].unique()
    preds_df = pd.DataFrame()
    for index, c in enumerate(city_code_list):
        city_preds = pd.read_csv(os.path.join('../../prediction_results/multi_process_result/ResidualCityAddTrend','trend_city_resudial_prediction_12-15-22_GLUPoint4layerMultiProcessResidualCityAddTrend_Train_199_{}.csv'.format(c)), header=None)
        preds_df = pd.concat((preds_df, city_preds), axis=0, ignore_index=True)
   
    preds_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
    preds_df = preds_df.sort_values(by=['date_dt'])
    preds_df.to_csv('../../prediction_results/multi_process_result/ResidualCityAddTrend_prediction.csv', index=False, header = False)

if __name__ == '__main__':
    city_preds_merge()