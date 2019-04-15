import numpy as np 
import pandas as pd 
import datetime
import pickle
import os 
import torch.utils.data as data
import torch 
import random
from statsmodels.tsa.seasonal import seasonal_decompose 

from core.config import PreliminaryConfig as cfg
from utils.log import log

seed = cfg.TRAIN.seed#87#14
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class TrainDatasetLoader(data.Dataset):
    def __init__(self, transition_x, flow_x, flow_y):
        """
        Params:
            transition_x: [Samples x N x N x T x 1]
            flow_x: [Samples x N x T x H]
            flow_y: [Samples x N x 15 x 3]
        """
        # print(transition_x.shape)
        self.transition_x = transition_x
        # self.transition_x = np.transpose(transition_x, axes=[0,4,1,2,3])
        flow_x = np.reshape(flow_x, newshape=(flow_x.shape[0]*flow_x.shape[1], flow_x.shape[2], flow_x.shape[3]))
        flow_y = np.reshape(flow_y, newshape=(flow_y.shape[0]*flow_y.shape[1], flow_y.shape[2], flow_y.shape[3]))
        self.flow_x = np.transpose(flow_x[:, np.newaxis,:,:], axes=[0,3,1,2])
        self.flow_y = np.transpose(flow_y[:, np.newaxis,:,:], axes=[0,3,1,2])

        # flow_data = np.transpose(flow_x, axes=[0,3,1,2])
        # self.flow_x = flow_data[:,:,:,:]
        # self.flow_y = np.transpose(flow_y, axes=[0,3,1,2])
        days = np.arange(30)
        weeks = np.arange(4)
        last_days = np.arange(5)
        days_lambda = (np.exp(-1*0.05*days) + 1.0)/2.0 
        week_lambda = np.exp(-1*0.1*weeks) * 0.1
        last_days_lambda = np.exp(-1*0.1*last_days) * 0.08
        days_lambda = days_lambda[::-1]
        days_lambda[2::7] += week_lambda[::-1]
        days_lambda[1::7] += last_days_lambda[::-1]
        self.days_weights = np.array(days_lambda[np.newaxis, np.newaxis])
        #  = np.reshape( np.tile(days_lambda, reps=cfg.TRAIN.bs), newshape=(cfg.TRAIN.bs, 1, 1, cfg.DATASET.T))
    
    def __len__(self):
        return self.flow_x.shape[0]
    
    def __getitem__(self, index):
        if not cfg.DATASET.weighted:
            return self.transition_x[1], self.flow_x[index], self.flow_y[index]
        else:
            weighted_flow_x = np.append(self.flow_x[index], self.days_weights, axis=0)
            return self.transition_x[1], weighted_flow_x, self.flow_y[index]
        # return self.transition_x[index], self.flow_x[index], self.flow_y[index]
    
class TestDatasetLoader(data.Dataset):
    def __init__(self, transition_x, flow_x):
        """
        Params:
            transition_x: [Samples x N x N x T x 1]
            flow_x: [Samples x N x T x 3]
        """
        self.transition_x = transition_x#np.transpose(transition_x, axes=[0,4,1,2,3])
        flow_x = np.reshape(flow_x, newshape=(flow_x.shape[0]*flow_x.shape[1], flow_x.shape[2], flow_x.shape[3]))
        flow_data = np.transpose(flow_x[:, np.newaxis, :, :], axes=[0,3,1,2])
        self.flow_x = flow_data[:,:,:,:]

        days = np.arange(30)
        weeks = np.arange(4)
        last_days = np.arange(5)
        days_lambda = (np.exp(-1*0.05*days) + 1.0)/2.0 
        week_lambda = np.exp(-1*0.1*weeks) * 0.1
        last_days_lambda = np.exp(-1*0.1*last_days) * 0.08
        days_lambda = days_lambda[::-1]
        days_lambda[2::7] += week_lambda[::-1]
        days_lambda[1::7] += last_days_lambda[::-1]
        self.days_weights = np.array(days_lambda[np.newaxis, np.newaxis])
    
    def __len__(self):
        return self.flow_x.shape[0]
    
    def __getitem__(self, index):
        # if cfg.DATASET.diff:
        #     return self.transition_x[index], self.flow_diff_x[index], self.flow_x[index,:,:,-1]
        # else:
        if not cfg.DATASET.weighted:
            return self.transition_x[1], self.flow_x[index]
        else:
            weighted_flow_x = np.append(self.flow_x[index], self.days_weights, axis=0)
            return self.transition_x[1], weighted_flow_x

class BiTrainDatasetLoader(data.Dataset):
    def __init__(self, fwflow_x, bwflow_x, flow_y):
        """
        Params:
            flow_x: [Samples x N x T x H]
            flow_y: [Samples x N x 15 x 3]
        """
        fwflow_x = np.reshape(fwflow_x, newshape=(fwflow_x.shape[0]*fwflow_x.shape[1], fwflow_x.shape[2], fwflow_x.shape[3]))
        bwflow_x = np.reshape(bwflow_x, newshape=(bwflow_x.shape[0]*bwflow_x.shape[1], bwflow_x.shape[2], bwflow_x.shape[3]))
        flow_y = np.reshape(flow_y, newshape=(flow_y.shape[0]*flow_y.shape[1], flow_y.shape[2], flow_y.shape[3]))
        self.fwflow_x = np.transpose(fwflow_x[:, np.newaxis,:,:], axes=[0,3,1,2])
        self.bwflow_x = np.transpose(bwflow_x[:, np.newaxis,:,:], axes=[0,3,1,2])
        self.flow_y = np.transpose(flow_y[:, np.newaxis,:,:], axes=[0,3,1,2])

        days = np.arange(30)
        weeks = np.arange(4)
        last_days = np.arange(5)
        days_lambda = (np.exp(-1*0.05*days) + 1.0)/2.0 
        week_lambda = np.exp(-1*0.1*weeks) * 0.1
        last_days_lambda = np.exp(-1*0.1*last_days) * 0.08
        days_lambda = days_lambda[::-1]
        days_lambda[2::7] += week_lambda[::-1]
        days_lambda[1::7] += last_days_lambda[::-1]
        self.days_weights = np.array(days_lambda[np.newaxis, np.newaxis])
    
    def __len__(self):
        return self.fwflow_x.shape[0]
    
    def __getitem__(self, index):
        if not cfg.DATASET.weighted:
            return self.fwflow_x[index], self.bwflow_x[index], self.flow_y[index]
        else:
            fwweighted_flow_x = np.append(self.fwflow_x[index], self.days_weights, axis=0)
            bwweighted_flow_x = np.append(self.bwflow_x[index], self.days_weights, axis=0)
            return fwweighted_flow_x, bwweighted_flow_x, self.flow_y[index]
    
class BiTestDatasetLoader(data.Dataset):
    def __init__(self, fwflow_x, bwflow_x):
        """
        Params:
            flow_x: [Samples x N x T x 3]
        """
        fwflow_x = np.reshape(fwflow_x, newshape=(fwflow_x.shape[0]*fwflow_x.shape[1], fwflow_x.shape[2], fwflow_x.shape[3]))
        bwflow_x = np.reshape(bwflow_x, newshape=(bwflow_x.shape[0]*bwflow_x.shape[1], bwflow_x.shape[2], bwflow_x.shape[3]))
        self.fwflow_x = np.transpose(fwflow_x[:, np.newaxis,:,:], axes=[0,3,1,2])
        self.bwflow_x = np.transpose(bwflow_x[:, np.newaxis,:,:], axes=[0,3,1,2])

        days = np.arange(30)
        weeks = np.arange(4)
        last_days = np.arange(5)
        days_lambda = (np.exp(-1*0.05*days) + 1.0)/2.0 
        week_lambda = np.exp(-1*0.1*weeks) * 0.1
        last_days_lambda = np.exp(-1*0.1*last_days) * 0.08
        days_lambda = days_lambda[::-1]
        days_lambda[2::7] += week_lambda[::-1]
        days_lambda[1::7] += last_days_lambda[::-1]
        self.days_weights = np.array(days_lambda[np.newaxis, np.newaxis])
    
    def __len__(self):
        return self.bwflow_x.shape[0]
    
    def __getitem__(self, index):
        if not cfg.DATASET.weighted:
            return self.fwflow_x[index], self.bwflow_x[index]
        else:
            fwweighted_flow_x = np.append(self.fwflow_x[index], self.days_weights, axis=0)
            bwweighted_flow_x = np.append(self.bwflow_x[index], self.days_weights, axis=0)
            return fwweighted_flow_x, bwweighted_flow_x

class PreliminaryDataGenerator:
    def __init__(self, flow_file, transfer_file, process_transition_file, t_period=30, n_days=274, predict_days=15, nodes=98, train_proportion=0.8, date_index=None):
        
        self.train_proportion = train_proportion
        self.predict_days = predict_days
        self.t_period = t_period
        self.n_days = n_days
        self.n_samples = self.n_days - (self.t_period + self.predict_days) + 1
        self.nodes  = nodes

        self.date_index = date_index
        self.flow_file = flow_file
        self.transfer_file = transfer_file
        self.process_transition_file = process_transition_file

        # flow file load
        flow_df = pd.read_csv(self.flow_file)
        self.flow_df = flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])

        # transition file load
        transition_df = pd.read_csv(self.transfer_file)
        self.transition_df = transition_df.sort_values(by=['o_city_code', 'o_district_code', 'date_dt'])
    
    def generate(self):
        
        self.district_code_list = self.flow_df['district_code'].unique()
        self.city_code_list = self.flow_df['city_code'].unique()

        self.district_num = len(self.district_code_list)
        
        self.flow_data = self.flow_generate(flow_transfer=cfg.DATASET.flow_transform, diff=cfg.DATASET.diff) # generate self.input_flow_data
        log.logger.info('flow_data shape:{}'.format(self.flow_data.shape))
        self.transition_data = self.transition_generate() # generate self.input_transition_data
        
        if cfg.DATASET.trend:
            self.trend_feature_amplify()
        if cfg.DATASET.residual:
            self.residual_feature_amplify()

        ret = self.dataset_generate()

        return ret

    def trend_feature_amplify(self):

        trend_feature7 = np.zeros_like(self.flow_data[:,:,:3])
        for n in range(self.flow_data.shape[0]):
            trend_feature7[n,:,:] = seasonal_decompose(self.flow_data[n,:,:3], freq=7).trend # N x D x 3
            trend_feature7[n,:3,:] = trend_feature7[n,3:6,:]
            trend_feature7[n,-3:,:] = trend_feature7[n,-6:-3,:]

        self.flow_data = np.append(self.flow_data, trend_feature7, axis=-1) # N x D x 6

    def residual_feature_amplify(self):
        residual_feature7 = np.zeros_like(self.flow_data[:,:,:3])
        for n in range(self.flow_data.shape[0]):
            residual_feature7[n,:,:] = seasonal_decompose(self.flow_data[n,:,:3], freq=7).resid # N x D x 3
            residual_feature7[n,:3,:] = residual_feature7[n,3:6,:]
            residual_feature7[n,-3:,:] = residual_feature7[n,-6:-3,:]

        self.flow_data = np.append(self.flow_data, residual_feature7, axis=-1) # N x D x 6

    def flow_generate(self, flow_transfer=False, diff=False):
        """
        Params:
            flow_transfer: whether transfer flow data by ' log(x+1) '
            diff: whether add diff feature
        Output:
            flow_data: [N x days x H]
        """

        if flow_transfer:
            print(flow_transfer)
            self.flow_df[['dwell', 'flow_in', 'flow_out']] = np.log(1+self.flow_df[['dwell', 'flow_in', 'flow_out']])

        if not diff:
            flow_data = np.zeros(shape=(self.district_num, self.n_days, 3))
        else:
            flow_data = np.zeros(shape=(self.district_num, self.n_days, 6))
            self.flow_df[['dwell_diff', 'flow_in_diff', 'flow_out_diff']] = self.flow_df[['dwell', 'flow_in', 'flow_out']].diff(1)

        for index,district_node in enumerate(self.district_code_list):
            district_df = self.flow_df[self.flow_df['district_code'] == district_node]
            if diff:
                district_df.index = np.arange(len(district_df))
                district_df.loc[0,['dwell_diff', 'flow_in_diff', 'flow_out_diff']] = district_df.loc[1, ['dwell_diff', 'flow_in_diff','flow_out_diff']] # repalce nan
                flow_data[index, :, :] = district_df[['dwell', 'flow_in', 'flow_out', 'dwell_diff', 'flow_in_diff', 'flow_out_diff']].values # T x 6
            else:
                flow_data[index, :, :] = district_df[['dwell', 'flow_in', 'flow_out']].values # T x 3
        
        return flow_data

    def transition_generate(self):
        """
        Output:
            transition_data: [N x N x days x 1]
        """
        if not os.path.exists(self.process_transition_file):
            print('transition_data is not exist, and is generating now...')
            transition_data = np.zeros(shape=(self.district_num, self.district_num, self.n_days, 1))
            start_day = datetime.datetime.strptime('20170601', '%Y%m%d')
            def transition_assign_apply(df, oindex, dindex):
                date = df['date_dt']
                cnt = df['cnt']
                dd = datetime.datetime.strptime(str(date), '%Y%m%d')
                index = (dd - start_day).days
                transition_data[oindex, dindex, index, 0] = cnt

            for o_index, o_district_node in enumerate(self.district_code_list):
                for d_index, d_district_node in enumerate(self.district_code_list):
                    if o_index == d_index:
                        continue
                    district_transition_df = self.transition_df[(self.transition_df['o_district_code'] == o_district_node) &
                                    (self.transition_df['d_district_code'] == d_district_node)]
                    district_transition_df.apply(transition_assign_apply, axis=1, oindex=o_index, dindex=d_index)

            district_transition_sum_data = np.sum(transition_data, axis=1, keepdims=True) # district_num x 1 x n_days x 1
            transition_data = transition_data / district_transition_sum_data
            with open(self.process_transition_file, 'wb') as file:
                pickle.dump(transition_data, file)
        else:
            with open(self.process_transition_file, 'rb') as file:
                transition_data = pickle.load(file)     
        eye = np.eye(cfg.DATASET.nodes)   
        return transition_data + eye[:,:,np.newaxis,np.newaxis]
    
    def date_feature_amplify(self):
        """
            Add date one-hot feature.
        """
        nodes = self.flow_data.shape[0]
        n_days = self.flow_data.shape[1]
        date_features = np.zeros((nodes, n_days, 7))
        assign_one_hot = np.zeros((7, 7))
        for d in range(7):
            assign_one_hot[d,d] = 1

        for d in range(n_days):
            date_features[:,d,:] = assign_one_hot[d%7]

        self.flow_data = np.append(self.flow_data, date_features, axis=-1)

    def dataset_generate(self):

        # input prepare
        seq_len = self.t_period + self.predict_days
        self.input_flow_data = np.zeros(shape=(self.n_samples, self.district_num, seq_len, self.flow_data.shape[-1]))
        self.input_transition_data = np.zeros(shape=(self.n_samples, self.district_num, self.district_num, self.t_period, self.transition_data.shape[-1]))

        for i in range(self.n_samples):
            self.input_flow_data[i,:,:,:] = self.flow_data[:,i:i+seq_len,:]
            self.input_transition_data[i,:,:,:,:] = self.transition_data[:,:,i:i+self.t_period,:]

        train_num = int(self.train_proportion * self.n_samples)
        val_num = self.n_samples - train_num
        log.logger.info('Train number: {} , Val number: {}'.format(train_num, val_num))

        train_transition = self.input_transition_data[:train_num, :, :, :, :]
        val_transition = self.input_transition_data[train_num:, :, :, :, :]
        test_transition = self.transition_data[np.newaxis, :, :, -1*self.t_period:, :]# 1 x N x N x T x 1
        
        train_flow_x = self.input_flow_data[:train_num, :, :self.t_period, :]
        val_flow_x = self.input_flow_data[train_num:, :, :self.t_period, :]
        test_flow_x = self.flow_data[np.newaxis, :, -1*self.t_period:, :].copy() # 1 x N x T x 3

        if self.date_index == None:
            train_flow_y = self.input_flow_data[:train_num, :, self.t_period:, :3]
            val_flow_y = self.input_flow_data[train_num:, :, self.t_period:, :3]# n_samples x N x T x 3
        else:
            train_flow_y = self.input_flow_data[:train_num, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
            val_flow_y = self.input_flow_data[train_num:, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
            print(train_flow_y.shape, val_flow_y.shape)

        # print(self.flow_data.shape)
        self.mean_flow = np.mean(self.flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)# 1 x 1 x 1 x 3
        self.std_flow = np.std(self.flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)
        # print(self.mean_flow.shape, self.std_flow.shape)

        train_flow_x[:,:,:,:3] = Z_Score(train_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        val_flow_x[:,:,:,:3] = Z_Score(val_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        test_flow_x[:,:,:,:3] = Z_Score(test_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)

        # trend_feature normalize
        if cfg.DATASET.trend :
            train_flow_x[:,:,:,3:6] = Z_Score(train_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            val_flow_x[:,:,:,3:6] = Z_Score(val_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            test_flow_x[:,:,:,3:6] = Z_Score(test_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)

        if cfg.DATASET.residual:
            train_flow_x[:,:,:,6:] = Z_Score(train_flow_x[:,:,:,6:], self.mean_flow, self.std_flow)
            val_flow_x[:,:,:,6:] = Z_Score(val_flow_x[:,:,:,6:], self.mean_flow, self.std_flow)
            test_flow_x[:,:,:,6:] = Z_Score(test_flow_x[:,:,:,6:], self.mean_flow, self.std_flow)

        log.logger.info('Train Dataset shape:{},{},{}'.format(train_transition.shape, train_flow_x.shape, train_flow_y.shape))
        log.logger.info('Val Dataset shape:{},{},{}'.format(val_transition.shape, val_flow_x.shape, val_flow_y.shape))
        log.logger.info('Test Dataset shape:{},{}'.format(test_transition.shape, test_flow_x.shape, ))

        return train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, test_flow_x

    def test_date_generate(self):
        date_list = list()
        init_date = datetime.date(2018,3,2)
        for delta in range(cfg.DATASET.result_days):
            _date = init_date + datetime.timedelta(days = delta)
            date_list.append(_date.strftime('%Y%m%d'))
        return date_list

    def district_city_map(self):
        location_map = dict()

        for district_code in self.district_code_list:
            district_df = self.flow_df[self.flow_df['district_code'] == district_code]
            location_map[district_code] = district_df['city_code'].values[0]

        return location_map
    
    def city_district_map(self):
        city_district_dict = dict()
        city_district_index_dict = dict()
        for i in range(len(self.city_code_list)):
            city_district_dict[self.city_code_list[i]] = []
            city_district_index_dict[self.city_code_list[i]] = []

        for index,district_code in enumerate(self.district_code_list):
            district_df = self.flow_df[self.flow_df['district_code'] == district_code]
            city_district_dict[district_df['city_code'].values[0]].append(district_code)
            city_district_index_dict[district_df['city_code'].values[0]].append(index)

        return city_district_dict, city_district_index_dict

    def city_dataset_generate(self, city):

        #flow prepare
        city_district_dict = self.city_district_map()
        city_district_code_list = city_district_dict[city]
        city_district_num = len(city_district_code_list)
        flow_df = self.flow_df

        flow_df[['dwell', 'flow_in', 'flow_out']] = np.log(1+flow_df[['dwell', 'flow_in', 'flow_out']])

        city_flow_data = np.zeros(shape=(city_district_num, self.n_days, 6))

        for index,district_node in enumerate(city_district_code_list):
            district_df = flow_df[flow_df['district_code'] == district_node]
            city_flow_data[index, :, :] = district_df[['dwell', 'flow_in', 'flow_out']].values # T x 3


        # transition prepare(shuffle)
        with open(self.process_transition_file, 'rb') as file:
                transition_data = pickle.load(file)     
        eye = np.eye(cfg.DATASET.nodes)   
        transition_data =  transition_data + eye[:,:,np.newaxis,np.newaxis]
        transition_data = transition_data[:city_district_num,:city_district_num,:,:]

        # generate city dataset
        seq_len = self.t_period + self.predict_days
        city_input_flow_data = np.zeros(shape=(self.n_samples, city_district_num, seq_len, city_flow_data.shape[-1]))
        city_input_transition_data = np.zeros(shape=(self.n_samples, city_district_num, city_district_num, self.t_period, transition_data.shape[-1]))

        for i in range(self.n_samples):
            city_input_flow_data[i,:,:,:] = city_flow_data[:,i:i+seq_len,:]
            city_input_transition_data[i,:,:,:,:] = self.transition_data[:,:,i:i+self.t_period,:]

        train_num = int(self.train_proportion * self.n_samples)
        val_num = self.n_samples - train_num
        log.logger.info('Train number: {} , Val number: {}'.format(train_num, val_num))

        train_transition = city_input_transition_data[:train_num, :, :, :, :]
        val_transition = city_input_transition_data[train_num:, :, :, :, :]
        test_transition = transition_data[np.newaxis, :, :, -1*self.t_period:, :]# 1 x N x N x T x 1
        
        train_flow_x = city_input_flow_data[:train_num, :, :self.t_period, :]
        val_flow_x = city_input_flow_data[train_num:, :, :self.t_period, :]
        test_flow_x = city_flow_data[np.newaxis, :, -1*self.t_period:, :] # 1 x N x T x 3

        if self.date_index == None:
            train_flow_y = city_input_flow_data[:train_num, :, self.t_period:, :3]
            val_flow_y = city_input_flow_data[train_num:, :, self.t_period:, :3]# n_samples x N x T x 3
        else:
            train_flow_y = city_input_flow_data[:train_num, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
            val_flow_y = city_input_flow_data[train_num:, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
            print(train_flow_y.shape, val_flow_y.shape)
        # self.mean_flow = np.mean(self.input_flow_data[:,:,:,:3], axis=(0,1,2), keepdims=True)# 1 x N x 1 x 3
        # self.std_flow = np.std(self.input_flow_data[:,:,:,:3], axis=(0,1,2), keepdims=True)

        # print(self.flow_data.shape)
        city_mean_flow = np.mean(city_flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)# 1 x 1 x 1 x 3
        city_std_flow = np.std(city_flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)
        # print(self.mean_flow.shape, self.std_flow.shape)
        train_flow_x[:,:,:,:3] = Z_Score(train_flow_x[:,:,:,:3], city_mean_flow, city_std_flow)
        val_flow_x[:,:,:,:3] = Z_Score(val_flow_x[:,:,:,:3], city_mean_flow, city_std_flow)
        test_flow_x[:,:,:,:3] = Z_Score(test_flow_x[:,:,:,:3], city_mean_flow, city_std_flow)

        # train_flow_y[:,:,:,:3] = Z_Score(train_flow_y[:,:,:,:3], self.mean_flow, self.std_flow)
        # val_flow_y[:,:,:,:3] = Z_Score(val_flow_y[:,:,:,:3], self.mean_flow, self.std_flow)
        log.logger.info('City {}: '.format(city))
        log.logger.info('Train Dataset shape:{},{},{}'.format(train_transition.shape, train_flow_x.shape, train_flow_y.shape))
        log.logger.info('Val Dataset shape:{},{},{}'.format(val_transition.shape, val_flow_x.shape, val_flow_y.shape))
        log.logger.info('Test Dataset shape:{},{}'.format(test_transition.shape, test_flow_x.shape, ))

        return train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, test_flow_x, city_mean_flow, city_std_flow

class DataGenerator:
    def __init__(self, front_flow_file, final_flow_file, transfer_file, process_transition_file, t_period=30, n_days=274, predict_days=15, nodes=98, train_proportion=0.8, date_index=None):
        
        self.train_proportion = train_proportion
        self.predict_days = predict_days
        self.t_period = t_period
        
        self.date_index = date_index

        self.front_flow_file = front_flow_file
        self.final_flow_file = final_flow_file

        # self.front_transfer_file = cfg.FILE.front_transfer_file
        # self.final_transfer_file = cfg.FILE.final_transfer_file
        # self.process_transition_file = process_transition_file

        # flow file load
        front_flow_df = pd.read_csv(self.front_flow_file)
        final_flow_df = pd.read_csv(self.final_flow_file)

        self.front_flow_df = front_flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])
        self.final_flow_df = final_flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])

        self.nodes = len(self.front_flow_df['district_code'].unique())
        
        # transition file load
        # front_transition_df = pd.read_csv(self.front_transfer_file)
        # self.front_transition_df = front_transition_df.sort_values(by=['o_city_code', 'o_district_code', 'date_dt'])
        # final_transition_df = pd.read_csv(self.final_transfer_file)
        # self.final_transition_df = final_transition_df.sort_values(by=['o_city_code', 'o_district_code', 'date_dt'])

        # self.process_final_transition_file = cfg.FILE.process_final_transition_file
        # self.process_front_transition_file = cfg.FILE.process_front_transition_file

    def forward_generate(self):
        
        self.district_code_list = self.front_flow_df['district_code'].unique()
        self.city_code_list = self.front_flow_df['city_code'].unique()

        self.district_num = len(self.district_code_list)
        
        self.front_days = len(self.front_flow_df['date_dt'].unique())
        self.final_days = len(self.final_flow_df['date_dt'].unique())

        # N x D x H
        self.front_flow_data = self.flow_generate(self.front_flow_df, self.front_days, flow_transfer=cfg.DATASET.flow_transform, diff=cfg.DATASET.diff) # generate self.input_flow_data
        self.final_flow_data = self.flow_generate(self.final_flow_df, self.final_days, flow_transfer=cfg.DATASET.flow_transform, diff=cfg.DATASET.diff) # generate self.input_flow_data
        
        log.logger.info('front_flow_data shape:{}, final_flow_data shape:{}'.format(self.front_flow_data.shape, self.final_flow_data.shape))

        # self.front_transition_data = self.transition_generate(88,self.process_front_transition_file, self.front_transition_df, '20170523') # generate self.input_transition_data        
        # self.final_transition_data = self.transition_generate(73,self.process_final_transition_file, self.final_transition_df, '20170824') # generate self.input_transition_data


        total_flow_data = np.append(self.front_flow_data, self.final_flow_data, axis=1)
        self.mean_flow = np.mean(total_flow_data[:,:,:3], axis=(0,1), keepdims=True)
        self.std_flow = np.std(total_flow_data[:,:,:3], axis=(0,1), keepdims=True)# 1 x 1 x 3
        self.mean_flow = self.mean_flow[np.newaxis]
        self.std_flow = self.std_flow[np.newaxis]

        front_train_flow_x, front_train_flow_y, front_val_flow_x, front_val_flow_y = self.dataset_generate(self.front_flow_data.copy(), self.front_days)
        final_train_flow_x, final_train_flow_y, final_val_flow_x, final_val_flow_y = self.dataset_generate(self.final_flow_data.copy(), self.final_days)

        middle_test_flow_x = self.front_flow_data[np.newaxis, :, -1*self.t_period:, :].copy()
        final_test_flow_x = self.final_flow_data[np.newaxis, :, -1*self.t_period:, :].copy()
        
        middle_test_flow_x[:,:,:,:3] = Z_Score(middle_test_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        final_test_flow_x[:,:,:,:3] = Z_Score(final_test_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)

        # trend_feature normalize
        if cfg.DATASET.trend:
            middle_test_flow_x[:,:,:,3:6] = Z_Score(middle_test_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            final_test_flow_x[:,:,:,3:6] = Z_Score(final_test_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)

        if cfg.DATASET.season:
            middle_test_flow_x[:,:,:,6:] = Z_Score(middle_test_flow_x[:,:,:,6:], self.mean_flow, self.std_flow)
            final_test_flow_x[:,:,:,6:] = Z_Score(final_test_flow_x[:,:,:,6:], self.mean_flow, self.std_flow)

        if cfg.DATASET.residual:
            middle_test_flow_x[:,:,:,9:] =  Z_Score(middle_test_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)
            final_test_flow_x[:,:,:,9:] = Z_Score(final_test_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)

        if cfg.DATASET.front and cfg.DATASET.final:
            train_flow_x = np.append(front_train_flow_x, final_train_flow_x, axis=0)
            train_flow_y = np.append(front_train_flow_y, final_train_flow_y, axis=0)
            val_flow_x = np.append(front_val_flow_x, final_val_flow_x, axis=0)
            val_flow_y = np.append(front_val_flow_y, final_val_flow_y, axis=0)
        elif cfg.DATASET.front:
            train_flow_x = front_train_flow_x
            train_flow_y = front_train_flow_y
            val_flow_x = front_val_flow_x
            val_flow_y = front_val_flow_y
        elif cfg.DATASET.final:
            train_flow_x = final_train_flow_x
            train_flow_y = final_train_flow_y
            val_flow_x = final_val_flow_x
            val_flow_y = final_val_flow_y
        else:
            log.logger.info("Input params ERROR")
            exit(1)

        train_transition = np.zeros((train_flow_x.shape[0],2,2,2,2))
        val_transition = np.zeros((val_flow_x.shape[0],2,2,2,2))
        test_transition = np.zeros((2,2,2,2,2))#self.transition_dataset_generate()

        return train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, middle_test_flow_x, final_test_flow_x
    
    def backward_generate(self):
        """
        For middle days predict and train
        """
        self.district_code_list = self.front_flow_df['district_code'].unique()
        self.city_code_list = self.front_flow_df['city_code'].unique()

        self.district_num = len(self.district_code_list)

        self.front_days = len(self.front_flow_df['date_dt'].unique())
        self.final_days = len(self.final_flow_df['date_dt'].unique())

        self.front_flow_data = self.flow_generate( self.front_flow_df, self.front_days, flow_transfer=cfg.DATASET.flow_transform, diff=cfg.DATASET.diff) # generate self.input_flow_data
        self.final_flow_data = self.flow_generate( self.final_flow_df, self.final_days, flow_transfer=cfg.DATASET.flow_transform, diff=cfg.DATASET.diff) # generate self.input_flow_data
        
        self.reversed_front_flow_data = self.front_flow_data[:,::-1,:].copy()
        self.reversed_final_flow_data = self.final_flow_data[:,::-1,:].copy()
        log.logger.info('front_flow_data shape:{}, final_flow_data shape:{}'.format(self.reversed_front_flow_data.shape, self.reversed_final_flow_data.shape))
        # self.transition_data = self.transition_generate() # generate self.input_transition_data

        total_flow_data = np.append(self.reversed_front_flow_data.copy(), self.reversed_final_flow_data.copy(), axis=1)
        self.mean_flow = np.mean(total_flow_data[:,:,:3], axis=(0,1), keepdims=True)
        self.std_flow = np.std(total_flow_data[:,:,:3], axis=(0,1), keepdims=True)# 1 x 1 x 3
        self.mean_flow = self.mean_flow[np.newaxis]
        self.std_flow = self.std_flow[np.newaxis]

        front_train_flow_x, front_train_flow_y, front_val_flow_x, front_val_flow_y = self.dataset_generate(self.reversed_front_flow_data.copy(), self.front_days)
        final_train_flow_x, final_train_flow_y, final_val_flow_x, final_val_flow_y = self.dataset_generate(self.reversed_final_flow_data.copy(), self.final_days)

        middle_test_flow_x = self.reversed_final_flow_data[np.newaxis, :, -1*self.t_period:, :].copy()
        middle_test_flow_x[:,:,:,:3] = Z_Score(middle_test_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        # trend_feature normalize
        if cfg.DATASET.trend:
            middle_test_flow_x[:,:,:,3:6] = Z_Score(middle_test_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
        if cfg.DATASET.season:
            middle_test_flow_x[:,:,:,6:9] = Z_Score(middle_test_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)
        if cfg.DATASET.residual:
            middle_test_flow_x[:,:,:,9:] =  Z_Score(middle_test_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)

        if cfg.DATASET.front and cfg.DATASET.final:
            train_flow_x = np.append(front_train_flow_x, final_train_flow_x, axis=0)
            train_flow_y = np.append(front_train_flow_y, final_train_flow_y, axis=0)
            val_flow_x = np.append(front_val_flow_x, final_val_flow_x, axis=0)
            val_flow_y = np.append(front_val_flow_y, final_val_flow_y, axis=0)
        elif cfg.DATASET.front:
            train_flow_x = front_train_flow_x
            train_flow_y = front_train_flow_y
            val_flow_x = front_val_flow_x
            val_flow_y = front_val_flow_y
        elif cfg.DATASET.final:
            train_flow_x = final_train_flow_x
            train_flow_y = final_train_flow_y
            val_flow_x = final_val_flow_x
            val_flow_y = final_val_flow_y
        else:
            log.logger.info("Input params ERROR")
            exit(1)

        train_transition = np.zeros((train_flow_x.shape[0],2,2,2,2))
        val_transition = np.zeros((val_flow_x.shape[0],2,2,2,2))
        test_transition = np.zeros((2,2,2,2,2))#self.transition_dataset_generate()

        return train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, middle_test_flow_x

    def bigenerate(self):
        complete_flow_df = pd.read_csv(cfg.FILE.complete_flow_file)
        self.complete_flow_df = complete_flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])
        self.nodes = len(self.complete_flow_df['district_code'].unique())

        self.district_code_list = self.complete_flow_df['district_code'].unique()
        self.city_code_list = self.complete_flow_df['city_code'].unique()

        self.district_num = len(self.district_code_list)
        
        self.complete_days = len(self.complete_flow_df['date_dt'].unique())
        self.drop_days = np.arange(74,79)

        self.complete_flow_data = self.flow_generate(self.complete_flow_df.copy(), self.complete_days, flow_transfer=cfg.DATASET.flow_transform, diff=cfg.DATASET.diff) # generate self.input_flow_data
        log.logger.info('complete_flow_data shape:{}'.format(self.complete_flow_data.shape))# N x 166 x H
        
        self.mean_flow = np.mean(self.complete_flow_data[:,:,:3].copy(), axis=(0,1), keepdims=True)
        self.std_flow = np.std(self.complete_flow_data[:,:,:3].copy(), axis=(0,1), keepdims=True)# 1 x 1 x 3
        self.mean_flow = self.mean_flow[np.newaxis]
        self.std_flow = self.std_flow[np.newaxis]

        fwtrain_flow_x, bwtrain_flow_x, train_flow_y, fwval_flow_x, bwval_flow_x, val_flow_y, fwtest_flow_x, bwtest_flow_x = self.bidataset_generate(self.complete_flow_data.copy(), self.complete_days, self.drop_days)

        return fwtrain_flow_x, bwtrain_flow_x, train_flow_y, fwval_flow_x, bwval_flow_x, val_flow_y, fwtest_flow_x, bwtest_flow_x
    
    def city_generate(self, district_index_list):

        city_front_flow_data = self.front_flow_data[district_index_list]
        city_final_flow_data = self.final_flow_data[district_index_list]
        log.logger.info('city_front_flow_data shape:{}, city_final_flow_data shape:{}'.format(city_front_flow_data.shape, city_final_flow_data.shape))
        
        front_train_flow_x, front_train_flow_y, front_val_flow_x, front_val_flow_y = self.dataset_generate(city_front_flow_data.copy(), self.front_days)
        final_train_flow_x, final_train_flow_y, final_val_flow_x, final_val_flow_y = self.dataset_generate(city_final_flow_data.copy(), self.final_days)
        train_flow_x = np.append(front_train_flow_x, final_train_flow_x, axis=0)
        train_flow_y = np.append(front_train_flow_y, final_train_flow_y, axis=0)
        val_flow_x = np.append(front_val_flow_x, final_val_flow_x, axis=0)
        val_flow_y = np.append(front_val_flow_y, final_val_flow_y, axis=0)

        train_transition = np.zeros((train_flow_x.shape[0],2,2,2,2))
        val_transition = np.zeros((val_flow_x.shape[0],2,2,2,2))

        return train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y

    def trend_feature_amplify(self, flow_data):

        trend_feature7 = np.zeros_like(flow_data[:,:,:3])
        for n in range(flow_data.shape[0]):
            trend_feature7[n,:,:] = seasonal_decompose(flow_data[n,:,:3], freq=7).trend # N x D x 3
            trend_feature7[n,:3,:] = trend_feature7[n,3:6,:]
            trend_feature7[n,-3:,:] = trend_feature7[n,-6:-3,:]

        flow_data = np.append(flow_data, trend_feature7, axis=-1) # N x D x 6
        return flow_data

    def season_feature_amplify(self, flow_data):

        season_feature7 = np.zeros_like(flow_data[:,:,:3])
        for n in range(flow_data.shape[0]):
            season_feature7[n,:,:] = seasonal_decompose(flow_data[n,:,:3], freq=7).seasonal # N x D x 3

        flow_data = np.append(flow_data, season_feature7, axis=-1) # N x D x 6
        return flow_data

    def residual_feature_amplify(self, flow_data):
        residual_feature7 = np.zeros_like(flow_data[:,:,:3])
        for n in range(flow_data.shape[0]):
            residual_feature7[n,:,:] = seasonal_decompose(flow_data[n,:,:3], freq=7).resid # N x D x 3
            residual_feature7[n,:3,:] = residual_feature7[n,3:6,:]
            residual_feature7[n,-3:,:] = residual_feature7[n,-6:-3,:]

        flow_data = np.append(flow_data, residual_feature7, axis=-1) # N x D x 6
        return flow_data

    def flow_generate(self, flow_df, days, flow_transfer=False, diff=False):
        """
        Params:
            flow_transfer: whether transfer flow data by ' log(x+1) '
            diff: whether add diff feature
        Output:
            flow_data: [N x days x H]
        """

        if flow_transfer:
            print(flow_transfer)
            flow_df[['dwell', 'flow_in', 'flow_out']] = np.log(1+flow_df[['dwell', 'flow_in', 'flow_out']])

        if not diff:
            flow_data = np.zeros(shape=(self.district_num, days, 3))
        else:
            flow_data = np.zeros(shape=(self.district_num, days, 6))
            flow_df[['dwell_diff', 'flow_in_diff', 'flow_out_diff']] = flow_df[['dwell', 'flow_in', 'flow_out']].diff(1)

        for index,district_node in enumerate(self.district_code_list):
            district_df = flow_df[flow_df['district_code'] == district_node]
            if diff:
                district_df.index = np.arange(len(district_df))
                district_df.loc[0,['dwell_diff', 'flow_in_diff', 'flow_out_diff']] = district_df.loc[1, ['dwell_diff', 'flow_in_diff','flow_out_diff']] # repalce nan
                flow_data[index, :, :] = district_df[['dwell', 'flow_in', 'flow_out', 'dwell_diff', 'flow_in_diff', 'flow_out_diff']].values # T x 6
            else:
                flow_data[index, :, :] = district_df[['dwell', 'flow_in', 'flow_out']].values # T x 3
        
        if cfg.DATASET.trend:
            flow_data = self.trend_feature_amplify(flow_data)
        if cfg.DATASET.season:
            flow_data = self.season_feature_amplify(flow_data)
        if cfg.DATASET.residual:
            flow_data = self.residual_feature_amplify(flow_data)
        

        return flow_data

    def bidataset_generate(self, complete_data, complete_days, drop_days):

        district_num = complete_data.shape[0]
        seq_len = self.t_period * 2 + 7
        n_samples = complete_days - seq_len + 1 - len(drop_days)
        train_num = int(self.train_proportion * n_samples)

        # input prepare
        test_flow_data = np.zeros(shape=(1, district_num, seq_len+cfg.DATASET.middle_result_days, complete_data.shape[-1]))
        input_flow_data = np.zeros(shape=(n_samples, district_num, seq_len, complete_data.shape[-1]))
        index = 0
        for i in range(n_samples+len(drop_days)):
            if i not in drop_days:
                input_flow_data[index,:,:,:] = complete_data[:,i:i+seq_len,:]
                index += 1
            elif i == drop_days[0]:
                test_flow_data[0,:,:,:] = complete_data[:,i:i + seq_len + cfg.DATASET.middle_result_days,:]


        val_num = n_samples - train_num
        log.logger.info('Train number: {} , Val number: {}'.format(train_num, val_num))
        
        fwtrain_flow_x = input_flow_data[:train_num, :, :self.t_period, :]
        bwtrain_flow_x = input_flow_data[:train_num, :, -1*self.t_period-2:, :]
        
        fwval_flow_x = input_flow_data[train_num:, :, :self.t_period, :]
        bwval_flow_x = input_flow_data[train_num:, :, -1*self.t_period-2:, :]

        fwtest_flow_x = test_flow_data[:, :, :self.t_period, :]
        bwtest_flow_x = test_flow_data[:, :, -1*(self.t_period + 2 + cfg.DATASET.middle_result_days):, :]

        train_flow_y = input_flow_data[:train_num, :, self.t_period:1+self.t_period, :3]
        val_flow_y = input_flow_data[train_num:, :, self.t_period:1+self.t_period, :3]# n_samples x N x T x 3


        fwtrain_flow_x[:,:,:,:3] = Z_Score(fwtrain_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        fwval_flow_x[:,:,:,:3] = Z_Score(fwval_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        fwtest_flow_x[:,:,:,:3] = Z_Score(fwtest_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)

        bwtrain_flow_x[:,:,:,:3] = Z_Score(bwtrain_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        bwval_flow_x[:,:,:,:3] = Z_Score(bwval_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        bwtest_flow_x[:,:,:,:3] = Z_Score(bwtest_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        

        if cfg.DATASET.trend :
            fwtrain_flow_x[:,:,:,3:6] = Z_Score(fwtrain_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            bwtrain_flow_x[:,:,:,3:6] = Z_Score(bwtrain_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)

            fwval_flow_x[:,:,:,3:6] = Z_Score(fwval_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            bwval_flow_x[:,:,:,3:6] = Z_Score(bwval_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)

            fwtest_flow_x[:,:,:,3:6] = Z_Score(fwtest_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            bwtest_flow_x[:,:,:,3:6] = Z_Score(bwtest_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
        
        if cfg.DATASET.season :
            fwtrain_flow_x[:,:,:,6:9] = Z_Score(fwtrain_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)
            bwtrain_flow_x[:,:,:,6:9] = Z_Score(bwtrain_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)

            fwval_flow_x[:,:,:,6:9] = Z_Score(fwval_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)
            bwval_flow_x[:,:,:,6:9] = Z_Score(bwval_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)

            fwtest_flow_x[:,:,:,6:9] = Z_Score(fwtest_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)
            bwtest_flow_x[:,:,:,6:9] = Z_Score(bwtest_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)

        if cfg.DATASET.residual:
            fwtrain_flow_x[:,:,:,9:] = Z_Score(fwtrain_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)
            bwtrain_flow_x[:,:,:,9:] = Z_Score(bwtrain_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)

            fwval_flow_x[:,:,:,9:] = Z_Score(fwval_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)
            bwval_flow_x[:,:,:,9:] = Z_Score(bwval_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)

            fwtest_flow_x[:,:,:,9:] = Z_Score(fwtest_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)
            bwtest_flow_x[:,:,:,9:] = Z_Score(bwtest_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)

        log.logger.info('Train Flow Dataset shape:{},{},{}'.format(fwtrain_flow_x.shape, bwtrain_flow_x.shape, train_flow_y.shape))
        log.logger.info('Val Flow Dataset shape:{},{},{}'.format(fwval_flow_x.shape, bwval_flow_x.shape, val_flow_y.shape))
        log.logger.info('Test Flow Dataset shape:{},{}'.format(fwtest_flow_x.shape, bwtest_flow_x.shape))
        
        return fwtrain_flow_x, bwtrain_flow_x, train_flow_y, fwval_flow_x, bwval_flow_x, val_flow_y, fwtest_flow_x, bwtest_flow_x 

    def transition_generate(self, days, process_transition_file, transition_df, start_time):
        """
        Output:
            transition_data: [N x N x days x 1]
        """
        if not os.path.exists(process_transition_file):
            print('transition_data is not exist, and is generating now...')
            # middle
            days = 88
            transition_data = np.zeros(shape=(self.district_num, self.district_num, days, 1))
            start_day = datetime.datetime.strptime(start_time, '%Y%m%d')
            def transition_assign_apply(df, oindex, dindex):
                date = df['date_dt']
                cnt = df['cnt']
                dd = datetime.datetime.strptime(str(date), '%Y%m%d')
                index = (dd - start_day).days
                transition_data[oindex, dindex, index, 0] = cnt

            for o_index, o_district_node in enumerate(self.district_code_list):
                for d_index, d_district_node in enumerate(self.district_code_list):
                    if o_index == d_index:
                        continue
                    district_transition_df = transition_df[(transition_df['o_district_code'] == o_district_node) &
                                    (transition_df['d_district_code'] == d_district_node)]
                    district_transition_df.apply(transition_assign_apply, axis=1, oindex=o_index, dindex=d_index)

            district_transition_sum_data = np.sum(transition_data, axis=1, keepdims=True) # district_num x 1 x n_days x 1
            transition_data = transition_data / district_transition_sum_data
            with open(process_transition_file, 'wb') as file:
                pickle.dump(transition_data, file)
        else:
            with open(process_transition_file, 'rb') as file:
                transition_data = pickle.load(file)     
        eye = np.eye(cfg.DATASET.nodes)   
        return transition_data + eye[:,:,np.newaxis,np.newaxis]
    
    def date_feature_amplify(self, flow_data):
        """
            Add date one-hot feature.
        """
        nodes = flow_data.shape[0]
        n_days = flow_data.shape[1]
        date_features = np.zeros((nodes, n_days, 7))
        assign_one_hot = np.zeros((7, 7))
        for d in range(7):
            assign_one_hot[d,d] = 1

        for d in range(n_days):
            date_features[:,d,:] = assign_one_hot[d%7]

        self.flow_data = np.append(self.flow_data, date_features, axis=-1)

    # def transition_dataset_generate(self, n_samples, train_num):
        
    #     self.input_transition_data = np.zeros(shape=(n_samples, self.district_num, self.district_num, self.t_period, self.transition_data.shape[-1]))
    #     for i in range(n_samples):
    #         self.input_transition_data[i,:,:,:,:] = self.transition_data[:,:,i:i+self.t_period,:]

    #     train_transition = self.input_transition_data[:self.train_num, :, :, :, :]
    #     val_transition = self.input_transition_data[self.train_num:, :, :, :, :]
    #     test_transition = self.transition_data[np.newaxis, :, :, -1*self.t_period:, :]# 1 x N x N x T x 1


    #     log.logger.info('Train Transition Dataset shape:{}'.format(train_transition.shape))
    #     log.logger.info('Val Transition Dataset shape:{}'.format(val_transition.shape))
    #     return train_transition, val_transition, test_transition

    def dataset_generate(self, flow_data, days):
        """
        Params:
            flow_data: N x D x H
        """
        district_num = flow_data.shape[0]
        n_samples = days - (self.t_period + self.predict_days) + 1
        train_num = int(self.train_proportion * n_samples)

        # input prepare
        seq_len = self.t_period + self.predict_days
        input_flow_data = np.zeros(shape=(n_samples, district_num, seq_len, flow_data.shape[-1]))

        for i in range(n_samples):
            input_flow_data[i,:,:,:] = flow_data[:,i:i+seq_len,:]

        
        val_num = n_samples - train_num
        log.logger.info('Train number: {} , Val number: {}'.format(train_num, val_num))
        
        train_flow_x = input_flow_data[:train_num, :, :self.t_period, :]
        val_flow_x = input_flow_data[train_num:, :, :self.t_period, :]
        # test_flow_x = flow_data[np.newaxis, :, -1*self.t_period:, :].copy() # 1 x N x T x 3

        if self.date_index == None:
            train_flow_y = input_flow_data[:train_num, :, self.t_period:, :3]
            val_flow_y = input_flow_data[train_num:, :, self.t_period:, :3]# n_samples x N x T x 3
        else:
            train_flow_y = input_flow_data[:train_num, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
            val_flow_y = input_flow_data[train_num:, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
            print(train_flow_y.shape, val_flow_y.shape)

        # print(self.flow_data.shape)
        # self.mean_flow = np.mean(flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)# 1 x 1 x 1 x 3
        # self.std_flow = np.std(flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)
        # print(self.mean_flow.shape, self.std_flow.shape)

        train_flow_x[:,:,:,:3] = Z_Score(train_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        val_flow_x[:,:,:,:3] = Z_Score(val_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)
        # test_flow_x[:,:,:,:3] = Z_Score(test_flow_x[:,:,:,:3], self.mean_flow, self.std_flow)

        # trend_feature normalize
        if cfg.DATASET.trend :
            train_flow_x[:,:,:,3:6] = Z_Score(train_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            val_flow_x[:,:,:,3:6] = Z_Score(val_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
            # test_flow_x[:,:,:,3:6] = Z_Score(test_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)
        
        if cfg.DATASET.season :
            train_flow_x[:,:,:,6:9] = Z_Score(train_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)
            val_flow_x[:,:,:,6:9] = Z_Score(val_flow_x[:,:,:,6:9], self.mean_flow, self.std_flow)
            # test_flow_x[:,:,:,3:6] = Z_Score(test_flow_x[:,:,:,3:6], self.mean_flow, self.std_flow)

        if cfg.DATASET.residual:
            train_flow_x[:,:,:,9:] = Z_Score(train_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)
            val_flow_x[:,:,:,9:] = Z_Score(val_flow_x[:,:,:,9:], self.mean_flow, self.std_flow)
            # test_flow_x[:,:,:,6:] = Z_Score(test_flow_x[:,:,:,6:], self.mean_flow, self.std_flow)

        log.logger.info('Train Flow Dataset shape:{},{}'.format(train_flow_x.shape, train_flow_y.shape))
        log.logger.info('Val Flow Dataset shape:{},{}'.format(val_flow_x.shape, val_flow_y.shape))
        # log.logger.info('Test Dataset shape:{},{}'.format(test_transition.shape, test_flow_x.shape, ))

        return train_flow_x, train_flow_y, val_flow_x, val_flow_y#, test_transition, test_flow_x

    def test_date_generate(self):
        date_list = list()
        init_date = datetime.date(2018,3,2)
        for delta in range(cfg.DATASET.result_days):
            _date = init_date + datetime.timedelta(days = delta)
            date_list.append(_date.strftime('%Y%m%d'))
        return date_list

    def final_test_date_generate(self):
        middle_date_list = list()
        final_date_list = list()
        middle_init_date = datetime.date(2017,8,19)
        for delta in range(cfg.DATASET.middle_result_days):
            _date = middle_init_date + datetime.timedelta(days = delta)
            middle_date_list.append(_date.strftime('%Y%m%d'))
        
        final_init_date = datetime.date(2017,11,5)
        for delta in range(cfg.DATASET.final_result_days):
            _date = final_init_date + datetime.timedelta(days = delta)
            final_date_list.append(_date.strftime('%Y%m%d'))

        return middle_date_list, final_date_list

    def district_city_map(self):
        location_map = dict()

        for district_code in self.district_code_list:
            district_df = self.front_flow_df[self.front_flow_df['district_code'] == district_code]
            location_map[district_code] = district_df['city_code'].values[0]

        return location_map
    
    def city_district_map(self):
        city_district_dict = dict()
        city_district_index_dict = dict()
        for i in range(len(self.city_code_list)):
            city_district_dict[self.city_code_list[i]] = []
            city_district_index_dict[self.city_code_list[i]] = []

        for index,district_code in enumerate(self.district_code_list):
            district_df = self.front_flow_df[self.front_flow_df['district_code'] == district_code]
            city_district_dict[district_df['city_code'].values[0]].append(district_code)
            city_district_index_dict[district_df['city_code'].values[0]].append(index)

        return city_district_dict, city_district_index_dict

    # def city_dataset_generate(self, city, flow_df, days):
    #     #flow prepare
    #     city_district_dict = self.city_district_map()
    #     city_district_code_list = city_district_dict[city]
    #     city_district_num = len(city_district_code_list)


    #     flow_df[['dwell', 'flow_in', 'flow_out']] = np.log(1+flow_df[['dwell', 'flow_in', 'flow_out']])

    #     city_flow_data = np.zeros(shape=(city_district_num, days, 3))

    #     for index,district_node in enumerate(city_district_code_list):
    #         district_df = flow_df[flow_df['district_code'] == district_node]
    #         city_flow_data[index, :, :3] = district_df[['dwell', 'flow_in', 'flow_out']].values # T x 3


    #     # transition prepare(shuffle)
    #     with open(self.process_transition_file, 'rb') as file:
    #             transition_data = pickle.load(file)     
    #     eye = np.eye(cfg.DATASET.nodes)   
    #     transition_data =  transition_data + eye[:,:,np.newaxis,np.newaxis]
    #     transition_data = transition_data[:city_district_num,:city_district_num,:,:]

    #     # generate city dataset
    #     seq_len = self.t_period + self.predict_days
    #     city_input_flow_data = np.zeros(shape=(self.n_samples, city_district_num, seq_len, city_flow_data.shape[-1]))
    #     city_input_transition_data = np.zeros(shape=(self.n_samples, city_district_num, city_district_num, self.t_period, transition_data.shape[-1]))

    #     for i in range(self.n_samples):
    #         city_input_flow_data[i,:,:,:] = city_flow_data[:,i:i+seq_len,:]
    #         city_input_transition_data[i,:,:,:,:] = self.transition_data[:,:,i:i+self.t_period,:]

    #     train_num = int(self.train_proportion * self.n_samples)
    #     val_num = self.n_samples - train_num
    #     log.logger.info('Train number: {} , Val number: {}'.format(train_num, val_num))

    #     train_transition = city_input_transition_data[:train_num, :, :, :, :]
    #     val_transition = city_input_transition_data[train_num:, :, :, :, :]
    #     test_transition = transition_data[np.newaxis, :, :, -1*self.t_period:, :]# 1 x N x N x T x 1
        
    #     train_flow_x = city_input_flow_data[:train_num, :, :self.t_period, :]
    #     val_flow_x = city_input_flow_data[train_num:, :, :self.t_period, :]
    #     test_flow_x = city_flow_data[np.newaxis, :, -1*self.t_period:, :] # 1 x N x T x 3

    #     if self.date_index == None:
    #         train_flow_y = city_input_flow_data[:train_num, :, self.t_period:, :3]
    #         val_flow_y = city_input_flow_data[train_num:, :, self.t_period:, :3]# n_samples x N x T x 3
    #     else:
    #         train_flow_y = city_input_flow_data[:train_num, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
    #         val_flow_y = city_input_flow_data[train_num:, :, self.t_period+self.date_index:self.t_period+1+self.date_index, :3]
    #         print(train_flow_y.shape, val_flow_y.shape)

    #     city_mean_flow = np.mean(city_flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)# 1 x 1 x 1 x 3
    #     city_std_flow = np.std(city_flow_data[np.newaxis,:,:,:3], axis=(0,1,2), keepdims=True)

    #     train_flow_x[:,:,:,:3] = Z_Score(train_flow_x[:,:,:,:3], city_mean_flow, city_std_flow)
    #     val_flow_x[:,:,:,:3] = Z_Score(val_flow_x[:,:,:,:3], city_mean_flow, city_std_flow)
    #     test_flow_x[:,:,:,:3] = Z_Score(test_flow_x[:,:,:,:3], city_mean_flow, city_std_flow)

    #     log.logger.info('City {}: '.format(city))
    #     log.logger.info('Train Dataset shape:{},{},{}'.format(train_transition.shape, train_flow_x.shape, train_flow_y.shape))
    #     log.logger.info('Val Dataset shape:{},{},{}'.format(val_transition.shape, val_flow_x.shape, val_flow_y.shape))
    #     log.logger.info('Test Dataset shape:{},{}'.format(test_transition.shape, test_flow_x.shape, ))

    #     return train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, test_flow_x, city_mean_flow, city_std_flow
        

def Z_Score(x, mean, std):
    return (x - mean) / std

def Z_Inverse(x, mean, std):
    return x * std + mean


