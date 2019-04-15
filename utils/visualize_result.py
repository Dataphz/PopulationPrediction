import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd  
import os
import argparse  

def visualize_total_trend():
    plt.figure(1,figsize=(8,8))
    dwell_df = pd.DataFrame()
    flow_in_df = pd.DataFrame()
    flow_out_df = pd.DataFrame()

    os.chdir('../../prediction_results/tgcn')
    result = pd.read_csv('prediction_199_1321.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'GLU-1321dwell', 'GLU-1321flow_in', 'GLU-1321flow_out']
    groupby = result[['GLU-1321dwell', 'GLU-1321flow_in', 'GLU-1321flow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['GLU-1321dwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['GLU-1321flow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['GLU-1321flow_out']]), axis=1)

    os.chdir('../../prediction_results/arima_prediction_result')
    result = pd.read_csv('prediction.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'ARIMA-1264dwell', 'ARIMA-1264flow_in', 'ARIMA-1264flow_out']
    groupby = result[['ARIMA-1264dwell', 'ARIMA-1264flow_in', 'ARIMA-1264flow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['ARIMA-1264dwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['ARIMA-1264flow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['ARIMA-1264flow_out']]), axis=1)

    os.chdir('../../prediction_results/glu')

    # result = pd.read_csv('12-11-11_GLUPoint_Train_prediction_199.csv',header=None)
    result = pd.read_csv('12-12-15_GLUPoint_Finetune_prediction_79.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'GLUPoint-1755dwell', 'GLUPoint-1755flow_in', 'GLUPoint-1755flow_out']
    groupby = result[['GLUPoint-1755dwell', 'GLUPoint-1755flow_in', 'GLUPoint-1755flow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['GLUPoint-1755dwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['GLUPoint-1755flow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['GLUPoint-1755flow_out']]), axis=1)

    result = pd.read_csv('12-12-15_GLUPoint_Finetune_prediction_89.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'GLUPointBKdwell', 'GLUPointBKflow_in', 'GLUPointBKflow_out']
    groupby = result[['GLUPointBKdwell', 'GLUPointBKflow_in', 'GLUPointBKflow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['GLUPointBKdwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['GLUPointBKflow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['GLUPointBKflow_out']]), axis=1)

    result = pd.read_csv('12-11-21_GLUPoint_Train7x4withoutSigmoid_prediction_199-1251.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'GLUPoint1251dwell', 'GLUPoint1251flow_in', 'GLUPoint1251flow_out']
    groupby = result[['GLUPoint1251dwell', 'GLUPoint1251flow_in', 'GLUPoint1251flow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['GLUPoint1251dwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['GLUPoint1251flow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['GLUPoint1251flow_out']]), axis=1)

    # result = pd.read_csv('12-11-19_GLUPoint_TrainGooglewithoutsigmoid_prediction_199.csv',header=None)
    # result = pd.read_csv('12-11-20_GLUPoint_TrainGooglewithoutsigmoid1layer_prediction_199.csv',header=None)
    # result = pd.read_csv('12-11-20_GLUPoint_TrainGooglewithoutsigmoid2layer_prediction_199.csv',header=None)
    result = pd.read_csv('12-12-11_GLU5LayerPoint_Train_prediction_199_87-1214.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'GLUPoint1214dwell', 'GLUPoint1214flow_in', 'GLUPoint1214flow_out']
    groupby = result[['GLUPoint1214dwell', 'GLUPoint1214flow_in', 'GLUPoint1214flow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['GLUPoint1214dwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['GLUPoint1214flow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['GLUPoint1214flow_out']]), axis=1)

    os.chdir('../../prediction_results/fusion')
    result = pd.read_csv('Fusion_result_arima_glu4_glu5.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'fusion_dwell', 'fusion_flow_in', 'fusion_flow_out']
    groupby = result[['fusion_dwell', 'fusion_flow_in', 'fusion_flow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['fusion_dwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['fusion_flow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['fusion_flow_out']]), axis=1)

    

    os.chdir('../../prediction_results/fusion')
    result = pd.read_csv('Fusion_result_0.35_arima_0.65_glu5.csv',header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'fusion2_dwell', 'fusion2_flow_in', 'fusion2_flow_out']
    groupby = result[['fusion2_dwell', 'fusion2_flow_in', 'fusion2_flow_out']].groupby(result['date_dt']).sum()
    groupby.index = np.arange(len(groupby))
    dwell_df = pd.concat((dwell_df, groupby[['fusion2_dwell']]), axis=1)
    flow_in_df = pd.concat((flow_in_df, groupby[['fusion2_flow_in']]), axis=1)
    flow_out_df = pd.concat((flow_out_df, groupby[['fusion2_flow_out']]), axis=1)

    # ax=plt.subplot(311)
    # dwell_df[['ARIMA-1264dwell', 'GLU-1321dwell','GLUPoint-1755dwell','GLUPointBKdwell', 'GLUPoint1251dwell','GLUPointGoogledwell']].plot(ax=ax,title='dwell')
    # ax=plt.subplot(312)
    # flow_in_df[['ARIMA-1264flow_in', 'GLU-1321flow_in','GLUPoint-1755flow_in','GLUPointBKflow_in', 'GLUPoint1251flow_in','GLUPointGoogleflow_in']].plot(ax=ax,title='flow_in')
    # ax=plt.subplot(313)
    # flow_out_df[['ARIMA-1264flow_out', 'GLU-1321flow_out','GLUPoint-1755flow_out','GLUPointBKflow_out', 'GLUPoint1251flow_out','GLUPointGoogleflow_out']].plot(ax=ax,title='flow_out')
    # plt.show()
    ax=plt.subplot(311)
    dwell_df[['ARIMA-1264dwell', 'GLUPoint1251dwell','GLUPoint1214dwell','fusion_dwell', 'fusion2_dwell']].plot(ax=ax,title='dwell')
    ax=plt.subplot(312)
    flow_in_df[['ARIMA-1264flow_in', 'GLUPoint1251flow_in','GLUPoint1214flow_in','fusion_flow_in', 'fusion2_flow_in']].plot(ax=ax,title='flow_in')
    ax=plt.subplot(313)
    flow_out_df[['ARIMA-1264flow_out', 'GLUPoint1251flow_out','GLUPoint1214flow_out','fusion_flow_out','fusion2_flow_out']].plot(ax=ax,title='flow_out')
    plt.show()

def visualize_final_prediction_trend():
    viz_num = 9
    final_flow_df = pd.read_csv('../../data/final_flow_train.csv')
    district_code_values = final_flow_df['district_code'].unique()
    city_code_values = final_flow_df['city_code'].unique()


    # os.chdir('../../prediction_results/arima')
    # result_file = 'prediction_total.csv'
    # result_file = 'arima_final_prediction.csv'

    os.chdir('../../prediction_results/glu')
    # result_file = '12-15-21_GLUPoint4layer_Train_prediction_199_30150_87.csv' # 0.2601
    # result_file = '12-15-21_GLUPoint4layer1130_Train_prediction_199.csv'
    # result_file = '12-15-21_GLUPoint4layerAddTrend_Train_prediction_199.csv' # best 0.2589
    # result_file = '12-15-21_GLUPoint4layerAddTrendResidual_Train_prediction_199.csv'
    # os.chdir('../../prediction_results/multi_process_result/')
    # result_file = 'ResidualCityAddTrend_prediction.csv'

    # result_file = '12-16-20_GLUPointShuffleBatch_Train_prediction_199.csv' #0.2525
    # result_file = '12-16-23_GLUPointShuffleBatch21day3x7_Train_prediction_199.csv'#0.2567
    # result_file = '12-17-14_GLUPointShuffleBatch30day5layer_Train_prediction_199.csv'
    # result_file = '12-17-14_GLUPointShuffleBatch30day5layer_TrainonPretrain_prediction_199.csv'
    # result_file = '12-17-15_GLUPointShuffleBatch30day5layer_TrainALL_prediction_199.csv'
    # result_file = '12-17-15_GLUPointShuffleBatch30day5layer_Trainlrchange_prediction_149.csv'
    # result_file = '12-17-15_LSGLUPointShuffleBatch30day5layer_Train_prediction_199.csv'
    # Trend
    # result_file = '12-17-16_LSGLUPointShuffleBatch30day5layerTrend_Train_prediction_199.csv'# 0.2446
    # result_file = '12-17-20_GLUPointShuffleBatch30day5layerTrendWeightSeason_Train_prediction_199.csv' #0.2404
    # result_file = 'tws_city_point_prediction_12-17-23_GLUPointShuffleBatch30day5layerTrendWeightSeasonCityResidual_Train_199.csv'

    # drop holiday
    # result_file = '12-18-13_GLUPoint_Train_Smoothing_Drop_holiday_prediction_199.csv' 
    # result_file = '12-18-16_GLUPoint_Train_Smoothing_Drop_holiday_finalAugment_prediction_199.csv'
    # result_file = '12-18-16_GLUPoint_Train_Smoothing_Drop_holiday_finalAugment89_prediction_199.csv'
    # result_file = '12-18-23_GLUPoint_Train_Smoothing_Drop_holiday_finalandfront_prediction_199.csv'


    # result_file = '12-19-10_GLUPointTWS_Train_Smoothing_Drop_holiday_front_prediction_199.csv'
    # result_file = 'tws_city_point_prediction_12-19-10_GLUPointCityResidualTWS_Train_Smoothing_Drop_holiday_front_199.csv'

    # 28 days
    # result_file = '12-17-16_GLUPointShuffleBatch28day5layer_Train_prediction_199.csv'
    # 21 days 
    # result_file = '12-17-17_GLUPointShuffleBatch21day3layer_Train_prediction_199.csv'
    # result_file = '12-17-17_GLUPointShuffleBatch21day3layerTrend_Train_prediction_199.csv'

    # result_file = '12-16-17_GLUPointCity_Train_prediction_199.csv'
    os.chdir('../../prediction_results/fusion')
    # result_file = 'try95.csv'
    # result_file = 'fusion_bi.csv'
    result_file = 'fusion.csv'

    # result_file = 'Fusion_result_sm_fb.csv'
    # result_file = 'Fusion_result_cr+twsglu.csv'
    # result_file = 'Fusion_result_ar_glu4_mode12.csv'
    # os.chdir('../../prediction_results')
    # result_file = 'prediction_xgb.csv'
    
    


    pred_df = pd.read_csv(result_file,header=None)
    pred_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
    pred_df = pred_df.sort_values(by=['city_code','district_code','date_dt'])

    plt.ion()
    plt.figure(1,figsize=(11,8))

    for index, district_code in enumerate(district_code_values[90:]):
        final_district_df = final_flow_df[final_flow_df['district_code'] == district_code]
        district_df = pred_df[pred_df['district_code'] == district_code]
        title =  list(city_code_values).index(final_district_df['city_code'].values[0])
        ax=plt.subplot(331+(index%viz_num), title='{}_{}'.format(title,index))

        
        final_visual_df = final_district_df[-20:].copy() 
        final_visual_df[['dwell','flow_in','flow_out']] = final_visual_df[['dwell','flow_in','flow_out']]#np.log(final_visual_df[['dwell','flow_in','flow_out']] + 1)
        final_visual_df.index = np.arange(len(final_visual_df))
        final_visual_df[['dwell']].plot(ax=ax)
        # final_visual_df[['flow_in', 'flow_out']].plot(ax=ax)
        
        final_pred_visual_df = pd.concat((final_district_df[-1:], district_df[5:]), axis=0)
        final_pred_visual_df[['dwell','flow_in','flow_out']] = final_pred_visual_df[['dwell','flow_in','flow_out']]
        final_pred_visual_df.index = np.arange(len(final_visual_df)-1,len(final_visual_df)+10)
        # final_pred_visual_df[['flow_in', 'flow_out']].plot(ax=ax)
        final_pred_visual_df[['dwell']].plot(ax=ax)

        if (index+1) % viz_num==0:
            plt.show()
            line = input()
            while line!='g':
                if line == 'x':
                    plt.close()
                    exit(1)
                else:
                    line = input()
                    pass

            plt.close()
            plt.figure(1,figsize=(11,8))
    plt.show()

def visualize_middle_prediction_trend():
    viz_num = 9
    final_flow_df = pd.read_csv('../../data/final_flow_train.csv')
    # front_flow_df = pd.read_csv('../../data/front_flow_train.csv')
    front_flow_df = pd.read_csv('../../data/front_smoothing_drop_holiday_flow_train.csv')
    
    district_code_values = final_flow_df['district_code'].unique()
    city_code_values = final_flow_df['city_code'].unique()

    # os.chdir('../../prediction_results/arima')
    # result_file = 'prediction_total.csv'


    os.chdir('../../prediction_results/glu')
    # result_file = '12-17-23_BWGLUPointShuffleBatch30day5layerTrendWeightSeason_Train_middle_prediction_199.csv'
    # result_file = '12-15-21_GLUPoint4layerAddTrend_Train_prediction_199.csv' # best 0.2589
    # result_file = '12-15-21_GLUPoint4layer_Train_prediction_199_30150_87.csv' # 0.2601
    # result_file = '12-16-17_GLUPointCity_Train_prediction_199.csv'

    # result_file = '12-16-20_GLUPointShuffleBatch_Train_prediction_199.csv' #0.2525   
    # result_file = '12-18-09_GLUPoint_Train_Smoothing_prediction_199.csv' 
    # result_file = '12-17-14_GLUPointShuffleBatch30day5layer_Train_prediction_199.csv'
    # result_file = '12-17-15_LSGLUPointShuffleBatch30day5layer_Train_prediction_199.csv'

    # result_file = '12-18-13_GLUPoint_Train_Smoothing_Drop_holiday_prediction_199.csv'# 2325

    # result_file = '12-18-15_BWGLUPointTSW_Train_Smoothing_Drop_holiday_middle_prediction_199.csv'

    # augment 
    result_file = '12-18-16_GLUPoint_Train_Smoothing_Drop_holiday_finalAugment_prediction_199.csv'
    # result_file = '12-18-16_GLUPoint_Train_Smoothing_Drop_holiday_finalAugment89_prediction_199.csv'

    # result_file = '12-18-19_BiGLUPoint_Train_Smoothing_Drop_holiday_finalAugment82_prediction_199.csv'
    # result_file = '12-18-19_BiGLUPoint_Train_Smoothing_Drop_holiday_finalAugment82_ALLDataset_prediction_199.csv'
    # result_file = '12-19-02_BiGLUPoint21_Train_Smoothing_Drop_holiday_front_prediction_199.csv'
    # result_file ='12-19-02_BiGLUPoint21_71logtime_Train_Smoothing_Drop_holiday_front_prediction_199.csv'
    # result_file ='12-19-02_BiGLUPoint14_71logtime_Train_Smoothing_Drop_holiday_front_prediction_199.csv'
    # result_file = '12-19-03_BiGLUPoint14_71logtime_5days_Train_Smoothing_Drop_holiday_front_prediction_199.csv'
    # result_file = '12-19-09_BiGLUPoint71logtime_14+2_5days_Train_Smoothing_Drop_holiday_front_prediction_199.csv' #  有效果 应该最好
    # result_file = '12-19-09_BiGLUPoint71logtime_30+2_5days_Train_Smoothing_Drop_holiday_front_prediction_199.csv'
    # result_file  = '12-19-10_BiGLUPoint71logtime_14+2_5days_Check_prediction_199.csv'# 21 days 跟14天不好说，差挺大的。。得看线上成绩
    # result_file = '12-19-10_BiGLUPoint71logtime_14+2_5daysTrend_Train_Smoothing_Drop_holiday_front_prediction_199.csv'# 赌？
    # result_file = '12-19-10_BiGLUPoint71logtime_21+2_5daysTrend_Train_Smoothing_Drop_holiday_front_prediction_199.csv'

   
    # trend
    # result_file = '12-17-16_LSGLUPointShuffleBatch30day5layerTrend_Train_prediction_199.csv'# 0.2446
    # trend+weight 
    # result_file = '12-17-19_GLUPointShuffleBatch30day5layerWeighted_Train_prediction_199.csv'#0.2431

    # tsw with normal
    # result_file = '12-17-20_GLUPointShuffleBatch30day5layerTrendWeightSeason_Train_prediction_199.csv' #0.2404
    # only std
    # result_file = '12-17-20_GLUPointShuffleBatch30day5layerTrendWeightSeasononlystd_Train_prediction_199.csv'

    # weight
    # result_file = '12-17-18_GLUPointShuffleBatch30day5layerWeighted_Train_prediction_199.csv'
    #28
    # result_file = '12-17-16_GLUPointShuffleBatch28day5layerTrend_Train_prediction_199.csv'

    # result_file = '12-17-14_GLUPointShuffleBatch30day5layer_TrainonPretrain_prediction_199.csv'

    #on all
    # result_file = '12-17-12_GLUPointShuffleBatch30day5layerAll_Train_prediction_199.csv'

    # result_file = '12-16-23_GLUPointShuffleBatch21day3x7_Train_prediction_199.csv'#0.2567

    # backward middle predict
    # result_file = '12-17-21_BWGLUPointShuffleBatch30day5layerTrendWeightSeason_Train_middle_prediction_199.csv'

    # result_file = 'tws_city_point_prediction_12-17-23_GLUPointShuffleBatch30day5layerTrendWeightSeasonCityResidual_Train_199.csv'

    os.chdir('../../prediction_results/fusion')
    # result_file = 'try95.csv'
    # result_file = 'fusion_bitr.csv'
    result_file = 'fusion.csv'

    # result_file = 'Fusion_result_cr+twsglu+bw_weight.csv'
    # result_file = 'Fusion_result_cr+twsglu+bw_half.csv'#0.2366
    # result_file = 'Fusion_result_cr+twsglu.csv'
    # result_file = 'Fusion_result_sb+glu.csv'

    pred_df = pd.read_csv(result_file,header=None)
    pred_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
    pred_df = pred_df.sort_values(by=['city_code','district_code','date_dt'])

    plt.ion()
    plt.figure(1,figsize=(11,8))

    # visual_elements = [ 'flow_in', 'flow_out']
    visual_elements = [ 'dwell']
    for index, district_code in enumerate(district_code_values):
        
        front_district_df = front_flow_df[front_flow_df['district_code'] == district_code]
        district_df = pred_df[pred_df['district_code'] == district_code]
        final_district_df = final_flow_df[final_flow_df['district_code'] == district_code]
        title =  list(city_code_values).index(final_district_df['city_code'].values[0])
        ax=plt.subplot(331+(index%9), title=f'{title,index}')


        front_visual_df = front_district_df[-21:]
        front_visual_df[['dwell','flow_in','flow_out']] = np.log(front_visual_df[['dwell','flow_in','flow_out']] + 1)
        front_visual_df.index = np.arange(len(front_visual_df))
        # front_visual_df[['flow_in', 'flow_out']].plot(ax=ax)
        front_visual_df[visual_elements].plot(ax=ax)
        
        final_pred_visual_df = district_df[:5]
        final_pred_visual_df[['dwell','flow_in','flow_out']] = np.log(final_pred_visual_df[['dwell','flow_in','flow_out']] + 1)
        final_pred_visual_df.index = np.arange(len(front_visual_df),len(front_visual_df)+5)
        # final_pred_visual_df[['flow_in', 'flow_out']].plot(ax=ax)
        final_pred_visual_df[visual_elements].plot(ax=ax)

        final_visual_df = final_district_df[-14:]
        final_visual_df[['dwell','flow_in','flow_out']] = np.log(final_visual_df[['dwell','flow_in','flow_out']] + 1)
        final_visual_df.index = np.arange(len(front_visual_df)+5,len(front_visual_df)+5+len(final_visual_df))
        # final_visual_df.index = np.arange(len(final_visual_df))#
        # final_visual_df[['flow_in', 'flow_out']].plot(ax=ax)
        final_visual_df[visual_elements].plot(ax=ax)

        # final_pred_visual_df = district_df[5:]#pd.concat((final_district_df[-1:], ), axis=0)
        # final_pred_visual_df[['dwell','flow_in','flow_out']] = np.log(final_pred_visual_df[['dwell','flow_in','flow_out']] + 1)
        # # final_pred_visual_df.index = np.arange(len(final_visual_df)-1,len(final_visual_df)+10)
        # final_pred_visual_df.index = np.arange(len(front_visual_df)+5+len(final_visual_df),len(front_visual_df)+15+len(final_visual_df))
        # # final_pred_visual_df[['flow_in', 'flow_out']].plot(ax=ax)
        # final_pred_visual_df[visual_elements].plot(ax=ax)

        ax.legend().remove()
        if (index+1) % viz_num==0:
            plt.show()
            line = input()
            while line!='g':
                if line == 'x':
                    plt.close()
                    exit(1)
                else:
                    pass
                line = input()
            plt.close()
            plt.figure(1,figsize=(11,8))
    plt.show()

def preliminary_visualize_prediction_trend():
    viz_num = 9
    # flow_df = pd.read_csv('../dataset/data/flow_train.csv')
    flow_df = pd.read_csv('../../data/flow_train.csv')
    district_code_values = flow_df['district_code'].unique()


    # os.chdir('../../prediction_results/arima_prediction_result')
    # result_file = 'prediction.csv'

    os.chdir('../../prediction_results/glu')
    # add code
    # result_file = '12-12-23_GLUPointPeriod7_60_Train_prediction_199.csv'
    # result_file='12-12-13_GLUFusion_TrainAlongTest_prediction_199.csv'
    # result_file='12-12-11_GLU5LayerPoint_Train_prediction_199_87-1214.csv'
    # result_file = '12-13-14_GLUofDGLUCheck_Train_prediction_199.csv'
    # result_file='12-12-14_GLUPoint_Finetune_prediction_19.csv'
    # result_file='12-12-16_GLU_Train_prediction_249.csv'
    # result_file = '12-12-16_GLUPointMedium_Train_prediction_199.csv'
    # result_file = '12-12-17_GLUPointShort_Train_prediction_199.csv'
    # result_file = '12-13-00_GLUTWFusion_Train_prediction_199.csv'
    # result_file = '12-13-01_GLUClip0.1_Train_prediction_199.csv'
    # result_file = '12-13-15_DGLU7_Train_prediction_199.csv'
    # result_file = 'city_resudial_prediction_12-13-19_GLUPointCityResdial_Check_199.csv'
    # result_file = 'city_resudial_prediction_12-13-19_GLUPointCityResdial_Train_199.csv'# 1151
    # result_file = '12-14-12_GLUPointAddTrend_Train_prediction_199.csv'
    # result_file = '12-14-13_GLUPointAddTrendAddResidual_Train_prediction_199.csv'
    # result_file = '12-14-14_GLUPointCity_Train_prediction_199.csv'
    # result_file = 'city_resudial_prediction_12-14-15_GLUPointResidualCityAddTrend_Train_199.csv'
    # result_file = '12-15-00_GLUPoint_Train_prediction_199.csv'


    # os.chdir('../../prediction_results/fusion')
    # result_file = 'Fusion_result_double_0.65_arima_0.35_glu5_0.35_week4.csv'
    # result_file = 'Fusion_result_arima_glu4_glu5.csv'
    # result_file = 'Fusion_result_arima_glu4_glu5.csv'
    # result_file = 'Fusion_result_0.5_arima_0.2_glu5_0.3_week4.csv'
    # result_file = 'Fusion_result_0.5_arima_0.1_glu5_0.4_week4.csv'
    # result_file = 'Fusion_result_double_0.65_glu5_0.35_dglu7.csv'
    # result_file = 'Fusion_result_double_0.5_glu5_0.5_dglu7.csv'
    # result_file= 'Fusion_result_0.35_aima_0.65_cr.csv'

    result = pd.read_csv(result_file,header=None)
    result.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
    result = result.sort_values(by=['city_code','district_code','date_dt'])

    plt.ion()
    plt.figure(1,figsize=(11,8))
    district_code = district_code_values[12]
    district_df = flow_df[flow_df['district_code']==district_code]
    district_df.index=np.arange(len(district_df))

    # result_district_df = result[result['district_code']==district_code]
    # district_df = pd.concat((district_df[-14:], result_district_df), axis=0)
    # district_df.index=np.arange(len(district_df))
    # district_df[['dwell','flow_in','flow_out']].plot(title='lalala')
    # print(np.log(district_df['flow_in'].values+1))
    # plt.show()
    

    # print(district_df['dwell'].values)
    
    for index,district_code in enumerate(district_code_values):
        
        ax=plt.subplot(331+(index%9))
        district_df = flow_df[flow_df['district_code']==district_code]
        district_df.index=np.arange(len(district_df))
        
        result_district_df = result[result['district_code']==district_code]
        print(result_district_df.shape)
        district_df = pd.concat((district_df[-30:], result_district_df), axis=0)
        district_df.index=np.arange(len(district_df))
        district_df[['dwell','flow_in','flow_out']] = np.log(district_df[['dwell','flow_in','flow_out']] + 1)
        district_df[['dwell','flow_in','flow_out']].plot(ax=ax,title='{}'.format(index))
        if (index+1) % viz_num==0:
            plt.show()
            line = input()
            while line!='g':
                if line == 'x':
                    plt.close()
                    exit(1)
                else:
                    pass
            plt.close()
            plt.figure(1,figsize=(11,8))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='mode to visualize', default='0')
    parser.add_argument('--model', help='model to visualize', default='0')
    parser.add_argument('--pred', help='prediction csv', default='0')

    args = parser.parse_args()
    # path_configure = {'tgcn':'tgcn', 'glu':'glu','arima':'arima_prediction_result'}

    if args.mode == '0':
        visualize_total_trend()
    elif args.mode == '1':
        visualize_final_prediction_trend()
    elif args.mode == '2':
        visualize_middle_prediction_trend()
