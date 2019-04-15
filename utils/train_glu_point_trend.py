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
from model.glu import GLUModel, MediumGLUModel, ShortGLUModel, LongGLUModel, PeriodGLUModel
from dataset.data_utils import DataGenerator, TrainDatasetLoader, TestDatasetLoader, Z_Inverse
from model.loss import RMSLELoss, DiffRMSLELoss
from utils.log import log
from utils.predict_generate import *

def train_GLUPointTrend(args, date_index=None):

    seed = cfg.TRAIN.seed#87#14
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print('Dataset Loading...')
    st = time.clock()
    if date_index == None:
        dataset = DataGenerator(cfg.FILE.front_flow_file, cfg.FILE.final_flow_file, cfg.FILE.transfer_file, cfg.FILE.process_transition_file, t_period=cfg.DATASET.T, 
                    n_days=cfg.DATASET.days, predict_days=cfg.DATASET.predict_days, nodes=cfg.DATASET.nodes, train_proportion=cfg.DATASET.train_proportion)
    else:
        dataset = DataGenerator(cfg.FILE.front_flow_file, cfg.FILE.final_flow_file, cfg.FILE.transfer_file, cfg.FILE.process_transition_file, t_period=cfg.DATASET.T, 
                    n_days=cfg.DATASET.days, predict_days=cfg.DATASET.predict_days*(date_index+1), nodes=cfg.DATASET.nodes, train_proportion=cfg.DATASET.train_proportion, date_index=date_index)
    train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, middle_test_flow_x, final_test_flow_x = dataset.forward_generate()

    train_dataset = TrainDatasetLoader(train_transition, train_flow_x, train_flow_y)
    train_loader = DataLoader(dataset = train_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    val_dataset = TrainDatasetLoader(val_transition, val_flow_x, val_flow_y)
    val_loader = DataLoader(dataset = val_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    middle_test_dataset = TestDatasetLoader(test_transition, middle_test_flow_x)
    middle_test_loader = DataLoader(dataset = middle_test_dataset, batch_size=cfg.DATASET.nodes, num_workers=0, shuffle=False)
    final_test_dataset = TestDatasetLoader(test_transition, final_test_flow_x)
    final_test_loader = DataLoader(dataset = final_test_dataset, batch_size=cfg.DATASET.nodes, num_workers=0, shuffle=False)

    print('Dataset Loaded, using {}'.format(time.clock()-st))

    mean_flow = torch.DoubleTensor(np.transpose(dataset.mean_flow, axes=[0,3,1,2]))
    std_flow = torch.DoubleTensor(np.transpose(dataset.std_flow, axes=[0,3,1,2]))
    days_weighted = torch.DoubleTensor(np.repeat(train_dataset.days_weights[np.newaxis], repeats=cfg.DATASET.nodes, axis=0))

    c_in = train_flow_x.shape[-1]
    if cfg.DATASET.weighted:
        c_in += 1

    if cfg.DATASET.T%30==0 or cfg.DATASET.T == 28:
        if cfg.TRAIN.period:
            model = PeriodGLUModel(c_in, input_days=cfg.DATASET.T).cuda()
        else:
            print('--------------------------------GLU Model-------------------------------')
            model = GLUModel(c_in, input_days=cfg.DATASET.T).cuda()
    elif cfg.DATASET.T == 21:
        model = MediumGLUModel(c_in, input_days=cfg.DATASET.T).cuda()
    elif cfg.DATASET.T == 14:
        model = ShortGLUModel(c_in, input_days=cfg.DATASET.T).cuda()
    elif cfg.DATASET.T == 49:
        model = LongGLUModel(c_in, input_days=cfg.DATASET.T).cuda()

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
        middle_test_iter = iter(middle_test_loader)
        final_test_iter = iter(final_test_loader)

        val_len = len(val_iter)

        for i in range(cfg.LOG.log_time):
            input_transition_x, input_flow_x, input_flow_y = next(train_iter)
            transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
            flow_y.resize_(input_flow_y.size()).copy_(input_flow_y)

            preds = model(flow_x, transition_x, constant_attention=True)
            loss = criterion(preds, flow_y)

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

            preds = model(flow_x, transition_x, constant_attention=True)
            loss = criterion(preds, flow_y)

            val_loss_temp += loss.item()

        log.logger.info('Step:{}, Learning_rate:{:.6f}, train_loss:{:.4f}, val_loss:{:.4f}'.format(p, lr, \
                            train_loss_temp/cfg.LOG.log_time, val_loss_temp/(val_len+0.01)))
        train_loss_temp = 0.0
        val_loss_temp = 0.0

        if (p+1) % cfg.TRAIN.test_time == 0:
            # test
            if date_index == None:
                input_transition_x, input_flow_x = next(middle_test_iter)
                transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
                flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
                preds = model(flow_x, transition_x, constant_attention=True) # 1 x 3 x N X 1
                preds_input = torch.cat([(preds.cpu().double()-mean_flow)/std_flow, input_flow_x[:,3:6,:,cfg.DATASET.T-7:cfg.DATASET.T-6], ], dim=1)# 1 x H x N x 1

                for i in range(1, cfg.DATASET.middle_result_days):
                    input_flow_x = torch.cat([input_flow_x[:,:6,:,1:], preds_input], dim=-1)
                    input_flow_x = torch.cat([input_flow_x, days_weighted], dim=1) # bs x 7 x 1 x T

                    input_flow_x = input_flow_x.data
                    flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
                    preds = model(flow_x, transition_x, constant_attention=True)
                    preds_input = torch.cat([(preds.cpu().double()-mean_flow)/std_flow, input_flow_x[:,3:6,:,cfg.DATASET.T-7:cfg.DATASET.T-6], ], dim=1)
                # generate result  
                middle_preds = torch.cat([(input_flow_x[:,:3,:,-1*(cfg.DATASET.middle_result_days-1):] * std_flow) + mean_flow, preds.cpu().double()], dim=-1)# 1 x 3 x N X 15

                input_transition_x, input_flow_x = next(final_test_iter)
                transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
                flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
                preds = model(flow_x, transition_x, constant_attention=True) # 1 x 3 x N X 1
                preds_input = torch.cat([(preds.cpu().double()-mean_flow)/std_flow, input_flow_x[:,3:6,:,cfg.DATASET.T-7:cfg.DATASET.T-6], ], dim=1)# 1 x H x N x 1

                for i in range(1, cfg.DATASET.final_result_days):
                    input_flow_x = torch.cat([input_flow_x[:,:6,:,1:], preds_input], dim=-1)
                    input_flow_x = torch.cat([input_flow_x, days_weighted], dim=1) # bs x 7 x 1 x T

                    input_flow_x = input_flow_x.data
                    flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
                    preds = model(flow_x, transition_x, constant_attention=True)
                    preds_input = torch.cat([(preds.cpu().double()-mean_flow)/std_flow, input_flow_x[:,3:6,:,cfg.DATASET.T-7:cfg.DATASET.T-6], ], dim=1)
                # generate result  
                final_preds = torch.cat([(input_flow_x[:,:3,:,-1*(cfg.DATASET.final_result_days-1):] * std_flow) + mean_flow, preds.cpu().double()], dim=-1)# 1 x 3 x N X 15
            

                np_middle_preds = middle_preds.cpu().data.numpy()
                np_middle_preds = np.transpose(np_middle_preds, axes=[0,2,3,1])

                np_final_preds = final_preds.cpu().data.numpy()
                np_final_preds = np.transpose(np_final_preds, axes=[0,2,3,1])
                # print(np_middle_preds.shape, np_final_preds.shape)
                middle_date_list, final_date_list = dataset.final_test_date_generate()
                final_generate_result(np_middle_preds, np_final_preds, middle_date_list, final_date_list, p, dataset.district_code_list, dataset.district_city_map())

            # save model on 199
                if p == cfg.TRAIN.epochs-1:
                    torch.save(model.state_dict(), os.path.join(cfg.LOG.experiment_model_dir, 'epoch_199.model'))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.LOG.experiment_model_dir, '{}_epoch_{}.model'.format(date_index,p)))
        model.train()