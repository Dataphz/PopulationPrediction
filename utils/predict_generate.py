import pandas as pd 
import numpy as np 
import os

from core.config import PreliminaryConfig as cfg 

def backward_middle_generate_result(middle_preds, middle_date_list, p, district_code_list, location_map):

    preds_df = pd.DataFrame()
    # print(middle_preds.shape, final_preds.shape)
    for index,district_code in enumerate(district_code_list):
        
        tmp_df = pd.DataFrame(data=middle_date_list, columns=['date_dt'])
        tmp_df['city_code'] = location_map[district_code]
        tmp_df['district_code'] = district_code
        for i in range(3):
            district_element_preds = pd.Series(middle_preds[index,0,:,i])
            # district_element_preds = pd.Series(middle_preds[0,index,:,i])
            if cfg.DATASET.flow_transform:
                district_element_preds = np.exp(district_element_preds) - 1
            tmp_df = pd.concat([tmp_df, district_element_preds], axis=1)

        tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

    preds_df = preds_df.sort_values(by=['date_dt'])
    preds_df.to_csv(os.path.join(cfg.LOG.experiment_result_dir, '{}_prediction_{}.csv'.format(cfg.LOG.timestamp, p)), index=False, header = False)


def final_generate_result(middle_preds, final_preds, middle_date_list, final_date_list, p, district_code_list, location_map):

    preds_df = pd.DataFrame()
    # print(middle_preds.shape, final_preds.shape)
    for index,district_code in enumerate(district_code_list):
        
        tmp_df = pd.DataFrame(data=middle_date_list, columns=['date_dt'])
        tmp_df['city_code'] = location_map[district_code]
        tmp_df['district_code'] = district_code
        for i in range(3):
            district_element_preds = pd.Series(middle_preds[index,0,:,i])
            # district_element_preds = pd.Series(middle_preds[0,index,:,i])
            if cfg.DATASET.flow_transform:
                district_element_preds = np.exp(district_element_preds) - 1
            tmp_df = pd.concat([tmp_df, district_element_preds], axis=1)

        tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

    for index,district_code in enumerate(district_code_list):
        
        tmp_df = pd.DataFrame(data=final_date_list, columns=['date_dt'])
        tmp_df['city_code'] = location_map[district_code]
        tmp_df['district_code'] = district_code
        for i in range(3):
            # district_element_preds = pd.Series(final_preds[0,index,:,i])
            district_element_preds = pd.Series(final_preds[index,0,:,i])
            if cfg.DATASET.flow_transform:
                district_element_preds = np.exp(district_element_preds) - 1
            tmp_df = pd.concat([tmp_df, district_element_preds], axis=1)

        tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)


    preds_df = preds_df.sort_values(by=['date_dt'])
    preds_df.to_csv(os.path.join(cfg.LOG.experiment_result_dir, '{}_prediction_{}.csv'.format(cfg.LOG.timestamp, p)), index=False, header = False)

def generate_result(np_preds, p, date_list, district_code_list, location_map):
    
    preds_df = pd.DataFrame()
    for index,district_code in enumerate(district_code_list):
        tmp_df = pd.DataFrame(data=date_list, columns=['date_dt'])
        tmp_df['city_code'] = location_map[district_code]
        tmp_df['district_code'] = district_code
        for i in range(3):
            district_element_preds = pd.Series(np_preds[0,index,:,i])
            if cfg.DATASET.flow_transform:
                district_element_preds = np.exp(district_element_preds) - 1
            tmp_df = pd.concat([tmp_df, district_element_preds], axis=1)

        tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)
    preds_df = preds_df.sort_values(by=['date_dt'])
    preds_df.to_csv(os.path.join(cfg.LOG.experiment_result_dir, '{}_prediction_{}.csv'.format(cfg.LOG.timestamp, p)), index=False, header = False)

def generate_result_diff(np_preds, p, base_flow, date_list, district_code_list, location_map):
    """
        Diff np_preds.
    Params:
        np_preds: [1 x 3 x N x T]
        base_flow: flow_values of last day [1 x 3 x N]
    """
    np_preds = np.cumsum(np_preds, axis=-1) # along the T
    preds_df = pd.DataFrame()
    for index, district_code in enumerate(district_code_list):
        tmp_df = pd.DataFrame(data=date_list, columns=['date_dt'])
        tmp_df['city_code'] = location_map[district_code]
        tmp_df['district_code'] = district_code
        
        for i in range(3):
            district_element_preds = pd.Series(np_preds[0, index, :, i] + base_flow[0, i, index]) 
            tmp_df = pd.concat([tmp_df, district_element_preds], axis=1)

        tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)
    preds_df = preds_df.sort_values(by=['date_dt'])
    preds_df.to_csv(os.path.join(cfg.LOG.experiment_result_dir, '{}_prediction_{}.csv'.format(cfg.LOG.timestamp, p)), index=False, header = False)

        