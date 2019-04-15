import pandas as pd 
import numpy as np 
import os  


def fusion():
    Fusion = pd.DataFrame()

    # os.chdir('../../prediction_results/arima')
    # arima_result = pd.read_csv('prediction_total.csv',header=None)
    # arima_result.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
    # arima_result = arima_result.sort_values(by=['date_dt', 'city_code', 'district_code'])
    # # print(arima_result.head(),arima_result['dwell'][0:5])
    # district_code_list = arima_result['district_code'].unique()
    
    # os.chdir('../../prediction_results/fusion')
    # ag_result = pd.read_csv('Fusion_result_sm_fb.csv',header=None)
    # ag_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # ag_result = ag_result.sort_values(by=['date_dt', 'city_code', 'district_code'])

    os.chdir('../../prediction_results/glu')

    # glu4_result = pd.read_csv('tws_city_point_prediction_12-17-23_GLUPointShuffleBatch30day5layerTrendWeightSeasonCityResidual_Train_199.csv',header=None)
    # glu4_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # glu4_result = glu4_result.sort_values(by=['date_dt', 'city_code', 'district_code'])
    # # print(glu4_result.head(),glu4_result['dwell'][0:5])

    # sbglu_result = pd.read_csv('12-16-20_GLUPointShuffleBatch_Train_prediction_199.csv',header=None)
    # sbglu_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # sbglu_result = sbglu_result.sort_values(by=['date_dt', 'city_code', 'district_code'])

    smgluresult = pd.read_csv('12-18-13_GLUPoint_Train_Smoothing_Drop_holiday_prediction_199.csv',header=None)
    smgluresult.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    smgluresult = smgluresult.sort_values(by=['date_dt', 'city_code', 'district_code'])
    

    bwglu_result = pd.read_csv('12-19-09_BiGLUPoint71logtime_14+2_5days_Train_Smoothing_Drop_holiday_front_prediction_199.csv',header=None)
    bwglu_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    bwglu_result = bwglu_result.sort_values(by=['date_dt', 'city_code', 'district_code'])

    # tglu_result = pd.read_csv('12-18-23_GLUPoint_Train_Smoothing_Drop_holiday_finalandfront_prediction_199.csv',header=None)
    # tglu_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # tglu_result = tglu_result.sort_values(by=['date_dt', 'city_code', 'district_code'])


    # twsglu_result = pd.read_csv('12-17-20_GLUPointShuffleBatch30day5layerTrendWeightSeason_Train_prediction_199.csv',header=None)
    # twsglu_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # twsglu_result = twsglu_result.sort_values(by=['date_dt', 'city_code', 'district_code'])

    # week4_result = pd.read_csv('12-12-22_GLUPointPeriod7_30_Train_prediction_199-1310.csv',header=None)
    # week4_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # week4_result = week4_result.sort_values(by=['date_dt', 'city_code', 'district_code'])
    
    # dglu7_result = pd.read_csv('12-13-15_DGLU7_Train_prediction_199.csv',header=None)
    # dglu7_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # dglu7_result = dglu7_result.sort_values(by=['date_dt', 'city_code', 'district_code'])

    # cr_result = pd.read_csv('tws_city_point_prediction_12-19-10_GLUPointCityResidualTWS_Train_Smoothing_Drop_holiday_front_199.csv',header=None)
    # cr_result.columns = ['date_dt', 'city_code', 'district_code',  'dwell', 'flow_in', 'flow_out']
    # cr_result = cr_result.sort_values(by=['date_dt', 'city_code', 'district_code'])

    Fusion = pd.concat((Fusion, smgluresult[['date_dt', 'city_code', 'district_code','dwell','flow_in','flow_out']]), axis=0, ignore_index=True)
    ar = 0.35
    g5 = 0.65
    w4 = 0.35

    # NW = [12,14,19,33,30,27,24,57,60,67]
    NW2 = [171,173,174,177,179,180,181,182,183]
    NW1 = [85,87,88,90,91,92,93,95,96,97,98,100]

    for index, district_code in enumerate(district_code_list):
        # district_glu5_result = np.log(glu5_result[glu5_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        # district_glu4_result = np.log(glu4_result[glu4_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        # district_arima_result =np.log(arima_result[arima_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        district_smglu_result =np.log(smgluresult[smgluresult['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        district_bwglu_result =np.log(bwglu_result[bwglu_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        # district_week4_result =np.log(week4_result[week4_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        # district_dglu7_result =np.log(dglu7_result[dglu7_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        # district_ag_result =np.log(ag_result[ag_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)
        # district_cr_result =np.log(cr_result[cr_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values+1)

        # test
        # district_ag_result = ag_result[ag_result['district_code'] == district_code][['dwell','flow_in','flow_out']].values


        # if index in NW1 or index in NW2:
        weight = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        preds = np.exp(np.append( 0.5 * district_smglu_result[:5] + 0.5 * district_bwglu_result[:5],  district_smglu_result[5:], axis=0)) -1
        # print(district_smglu_result.shape)
        # preds = (np.append( 1.0 * district_ag_result[:5].copy(), 0.98 * district_ag_result[5:].copy(), axis=0))
        Fusion.loc[Fusion['district_code']==district_code, ['dwell','flow_in','flow_out']] = preds
        # else:
            # Fusion.loc[Fusion['district_code']==district_code, ['dwell','flow_in','flow_out']] = np.exp( np.append(district_glu4_result[:5], 1.0 * district_glu4_result[5:] + 0.0 * district_arima_result[5:], axis=0)) -1

        # Fusion = pd.concat((Fusion, district_week4_result[['dwell','flow_in','flow_out']]))
        # print(index, len(district_week4_result))



    # dwell = pd.Series((ar * np.log(arima_result['dwell'].values+1) + g5 * np.log(glu5_result['dwell'].values+1) + w4 * np.log(week4_result['dwell'].values+1)))#+ np.log(glu4_result['dwell'].values+1)
    # dwell = np.exp(dwell) - 1
    # flow_in = pd.Series((ar * np.log(arima_result['flow_in'].values+1) + g5 * np.log(glu5_result['flow_in'].values+1) + w4 * np.log(week4_result['flow_in'].values+1)))#+ np.log(glu4_result['flow_in'].values+1)
    # flow_in = np.exp(flow_in) - 1
    # flow_out = pd.Series((ar * np.log(arima_result['flow_out'].values+1) + g5 * np.log(glu5_result['flow_out'].values+1) + w4 * np.log(week4_result['flow_out'].values+1)))# + np.log(glu4_result['flow_out'].values+1)
    # flow_out = np.exp(flow_out) - 1

    # Fusion = pd.concat([Fusion, dwell], axis=1)
    # Fusion = pd.concat((Fusion, flow_in), axis=1)
    # Fusion = pd.concat((Fusion, flow_out), axis=1)
    Fusion = Fusion.sort_values(by=['date_dt'])
    # print(Fusion.head())
    # print(len(Fusion))
    Fusion.to_csv(os.path.join('../../prediction_results/fusion/result.csv'), index=False, header = False)

    # flow_out = (arima_result['flow_out'] + glu4_result['flow_out'] + glu5_result['flow_out']) / 3.0
    # dwell = (arima_result['dwell'] + glu4_result['dwell'] + glu5_result['dwell']) / 3.0
    
    

if __name__ == '__main__':
    fusion()

