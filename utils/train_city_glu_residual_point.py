
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from core.config import PreliminaryConfig as cfg 
from model.glu import GLUModel
from dataset.data_utils import DataGenerator, TrainDatasetLoader, TestDatasetLoader, Z_Inverse
from model.loss import RMSLELoss, DiffRMSLELoss
from utils.log import log
from utils.predict_generate import *

def train_city_redisual_models(args):
    seed = cfg.TRAIN.seed#87#14
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    st = time.clock()
    dataset = DataGenerator(cfg.FILE.front_flow_file, cfg.FILE.final_flow_file, cfg.FILE.transfer_file, cfg.FILE.process_transition_file, t_period=cfg.DATASET.T, 
                n_days=cfg.DATASET.days, predict_days=cfg.DATASET.predict_days, nodes=cfg.DATASET.nodes, train_proportion=cfg.DATASET.train_proportion)
    train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, middle_test_flow_x, final_test_flow_x = dataset.forward_generate()
    
    train_dataset = TrainDatasetLoader(train_transition, train_flow_x, train_flow_y)
    train_loader = DataLoader(dataset = train_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    val_dataset = TrainDatasetLoader(val_transition, val_flow_x, val_flow_y)
    val_loader = DataLoader(dataset = val_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    middle_test_dataset = TestDatasetLoader(test_transition, middle_test_flow_x)
    middle_test_loader = DataLoader(dataset = middle_test_dataset, batch_size=1, num_workers=0, shuffle=False)

    final_test_dataset = TestDatasetLoader(test_transition, final_test_flow_x)
    final_test_loader = DataLoader(dataset = final_test_dataset, batch_size=1, num_workers=0, shuffle=False)

    city_code_list = dataset.city_code_list
    city_district_map, city_district_index_map = dataset.city_district_map()
    middle_date_list, final_date_list = dataset.final_test_date_generate()

    mean_flow = torch.DoubleTensor(np.transpose(dataset.mean_flow, axes=[0,3,1,2]))
    std_flow = torch.DoubleTensor(np.transpose(dataset.std_flow, axes=[0,3,1,2]))
    flow_x = Variable(torch.FloatTensor(1).cuda())
    transition_x = Variable(torch.FloatTensor(1).cuda())

    base_model = GLUModel(c_in=3, input_days=cfg.DATASET.T).cuda()
    base_model.load_state_dict((torch.load(os.path.join(cfg.LOG.home_log_dir, '12-15-21_GLUPoint4layer_Train/model/epoch_199.model'))))

    p = 199
    
    preds_df = pd.DataFrame()
    base_model.eval()
    for i,c in enumerate(city_code_list):
        
        city_district_code_list = city_district_map[c] 
        city_district_index_list = city_district_index_map[c]
        city_district_num = len(city_district_code_list)

        train_city_residual_glu_model(base_model, train_loader, val_loader, c, city_district_num, city_district_index_list)
        
        # city prediction
        model = GLUModel(c_in=3, input_days=cfg.DATASET.T, nodes=city_district_num).cuda()
        model.load_state_dict(torch.load( os.path.join(cfg.LOG.experiment_model_dir, '{}_epoch_199.model'.format(c)) ))
        model.eval()

        # middle pred
        middle_test_iter = iter(middle_test_loader)
        input_transition_x, input_flow_x = next(middle_test_iter)
        transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
        flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
        # preds = model(flow_x, transition_x, constant_attention=True) # 1 x 3 x N X 1
        city_residual_preds = model(flow_x[:,:,city_district_index_list,:], transition_x, constant_attention=True)
        base_preds = base_model(flow_x, transition_x, constant_attention=True)
        base_preds[:,:,city_district_index_list,:] = base_preds[:,:,city_district_index_list,:] + city_residual_preds
        preds = base_preds# 1 x 3 x 98 x 1
        print(preds.shape)

        for i in range(1, cfg.DATASET.middle_result_days):
            input_flow_x = torch.cat([input_flow_x[:,:,:,1:], (preds.cpu().double()-mean_flow)/std_flow], dim=-1)
            input_flow_x = input_flow_x.data
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)

            city_residual_preds = model(flow_x[:,:,city_district_index_list,:], transition_x, constant_attention=True)
            base_preds = base_model(flow_x, transition_x, constant_attention=True)
            base_preds[:,:,city_district_index_list,:] = base_preds[:,:,city_district_index_list,:] + city_residual_preds
            log.logger.info('i:     {}'.format(city_residual_preds))
            preds = base_preds# 1 x 3 x 98 x 1

        middle_preds = torch.cat([(input_flow_x[:,:,:,-1*(cfg.DATASET.middle_result_days-1):] * std_flow) + mean_flow, preds.cpu().double()], dim=-1)# 1 x 3 x N X 15

        np_middle_preds = middle_preds.cpu().data.numpy()
        np_middle_preds = np.transpose(np_middle_preds, axes=[0,2,3,1])# 1 x 3 x 98 x 15
        print(np_middle_preds.shape)

        for index,district_code in enumerate(city_district_code_list):
            district_index = city_district_index_list[index]
            tmp_df = pd.DataFrame(data=middle_date_list, columns=['date_dt'])
            tmp_df['city_code'] = c
            tmp_df['district_code'] = district_code
            for i in range(3):
                district_element_preds = pd.Series(np_middle_preds[0,district_index,:,i])
                if cfg.DATASET.flow_transform:
                    district_element_preds = np.exp(district_element_preds) - 1
                tmp_df = pd.concat([tmp_df, district_element_preds], axis=1)

            tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
            preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

        
        # final pred
        final_test_iter = iter(final_test_loader)
        input_transition_x, input_flow_x = next(final_test_iter)
        transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
        flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
        city_residual_preds = model(flow_x[:,:,city_district_index_list,:], transition_x, constant_attention=True)
        base_preds = base_model(flow_x, transition_x, constant_attention=True)
        base_preds[:,:,city_district_index_list,:] = base_preds[:,:,city_district_index_list,:] + city_residual_preds
        preds = base_preds# 1 x 3 x 98 x 1
        print(preds.shape)

        for i in range(1, cfg.DATASET.final_result_days):
            input_flow_x = torch.cat([input_flow_x[:,:,:,1:], (preds.cpu().double()-mean_flow)/std_flow], dim=-1)
            input_flow_x = input_flow_x.data
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)

            city_residual_preds = model(flow_x[:,:,city_district_index_list,:], transition_x, constant_attention=True)
            base_preds = base_model(flow_x, transition_x, constant_attention=True)
            base_preds[:,:,city_district_index_list,:] = base_preds[:,:,city_district_index_list,:] + city_residual_preds
            log.logger.info('i:     {}'.format(city_residual_preds))
            preds = base_preds# 1 x 3 x 98 x 1

        final_preds = torch.cat([(input_flow_x[:,:,:,-1*(cfg.DATASET.final_result_days-1):] * std_flow) + mean_flow, preds.cpu().double()], dim=-1)# 1 x 3 x N X 15

        np_final_preds = final_preds.cpu().data.numpy()
        np_final_preds = np.transpose(np_final_preds, axes=[0,2,3,1])# 1 x 3 x 98 x 15
        print(np_final_preds.shape)

        for index,district_code in enumerate(city_district_code_list):
            district_index = city_district_index_list[index]
            tmp_df = pd.DataFrame(data=final_date_list, columns=['date_dt'])
            tmp_df['city_code'] = c
            tmp_df['district_code'] = district_code
            for i in range(3):
                district_element_preds = pd.Series(np_final_preds[0,district_index,:,i])
                if cfg.DATASET.flow_transform:
                    district_element_preds = np.exp(district_element_preds) - 1
                tmp_df = pd.concat([tmp_df, district_element_preds], axis=1)

            tmp_df.columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
            preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

    preds_df = preds_df.sort_values(by=['date_dt'])
    preds_df.to_csv(os.path.join(cfg.LOG.experiment_result_dir, 'city_resudial_prediction_{}_{}.csv'.format(cfg.LOG.timestamp, p)), index=False, header = False)

def train_city_residual_glu_model(base_model, train_loader, val_loader, city_code, city_district_num, city_district_code_list):
    seed = cfg.TRAIN.seed#87#14
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    base_model.eval()

    model = GLUModel(c_in=3, input_days=cfg.DATASET.T, nodes=city_district_num).cuda()
    params = []

    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value], 'lr':cfg.TRAIN.lr,\
                'weight_decay': 0}]
            else:
                params += [{'params':[value], 'lr':cfg.TRAIN.lr,\
                'weight_decay': cfg.TRAIN.weight_decay}]

    criterion = RMSLELoss().cuda()
    optimizer = torch.optim.SGD(params, momentum=0.9)

    flow_x = Variable(torch.FloatTensor(1).cuda())
    transition_x = Variable(torch.FloatTensor(1).cuda())
    flow_y = Variable(torch.FloatTensor(1).cuda())
    train_loss_temp = 0.0
    val_loss_temp = 0.0

    lr = cfg.TRAIN.lr
    model.train()

    for p in range(cfg.TRAIN.epochs):
        if p in cfg.TRAIN.lr_schedule:
            print('changing learning rate.....')
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * param_group['lr']
                lr = param_group['lr']
        # train 
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        val_len = len(val_iter)

        for i in range(cfg.LOG.log_time):
            input_transition_x, input_flow_x, input_flow_y = next(train_iter)
            transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
            flow_y.resize_(input_flow_y.size()).copy_(input_flow_y)

            base_preds = base_model(flow_x, transition_x, constant_attention=True)# bs x 3 x N x 1
            residual_preds = model(flow_x[:,:,city_district_code_list,:], transition_x, constant_attention=True)
            preds = residual_preds + base_preds[:,:,city_district_code_list,:].detach()
            loss = criterion(preds, flow_y[:,:,city_district_code_list,:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_temp += loss.item()

        model.eval()
        for i in range(val_len):
            input_transition_x, input_flow_x, input_flow_y = next(val_iter)
            transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
            flow_y.resize_(input_flow_y.size()).copy_(input_flow_y)

            base_preds = base_model(flow_x, transition_x, constant_attention=True)# bs x 3 x N x 1
            residual_preds = model(flow_x[:,:,city_district_code_list,:], transition_x, constant_attention=True)
            preds = residual_preds + base_preds[:,:,city_district_code_list,:].detach()
            loss = criterion(preds, flow_y[:,:,city_district_code_list,:])

            val_loss_temp += loss.item()

        log.logger.info('Step:{}, Learning_rate:{}, train_loss:{:.4f}, val_loss:{:.4f}'.format(p, lr, \
                            train_loss_temp/cfg.LOG.log_time, val_loss_temp/(val_len+0.01)))
        train_loss_temp = 0.0
        val_loss_temp = 0.0

        #save model
        if p == cfg.TRAIN.epochs-1:
            torch.save(model.state_dict(), os.path.join(cfg.LOG.experiment_model_dir, '{}_epoch_{}.model'.format(city_code, p)))
            
        model.train()