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
from model.glu import BiGLUModel, MediumBiGLUModel, ShortBiGLUModel
from dataset.data_utils import DataGenerator, BiTrainDatasetLoader, BiTestDatasetLoader, Z_Inverse
from model.loss import RMSLELoss, DiffRMSLELoss
from utils.log import log
from utils.predict_generate import *

def train_BiGLUPoint(args, date_index=None):

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
    fwtrain_flow_x, bwtrain_flow_x, train_flow_y, fwval_flow_x, bwval_flow_x, val_flow_y, fwtest_flow_x, bwtest_flow_x  = dataset.bigenerate()

    train_dataset = BiTrainDatasetLoader(fwtrain_flow_x, bwtrain_flow_x, train_flow_y)
    train_loader = DataLoader(dataset = train_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    val_dataset = BiTrainDatasetLoader(fwval_flow_x, bwval_flow_x, val_flow_y)
    val_loader = DataLoader(dataset = val_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    test_dataset = BiTestDatasetLoader(fwtest_flow_x, bwtest_flow_x)
    test_loader = DataLoader(dataset = test_dataset, batch_size=cfg.DATASET.nodes, num_workers=0, shuffle=False)

    print('Dataset Loaded, using {}'.format(time.clock()-st))

    mean_flow = torch.DoubleTensor(np.transpose(dataset.mean_flow, axes=[0,3,1,2]))
    std_flow = torch.DoubleTensor(np.transpose(dataset.std_flow, axes=[0,3,1,2]))

    c_in = fwtrain_flow_x.shape[-1]
    if cfg.DATASET.weighted:
        c_in += 1
        days_weighted = torch.DoubleTensor(np.repeat(train_dataset.days_weights[np.newaxis], repeats=cfg.DATASET.nodes, axis=0))

    if cfg.DATASET.T%30==0 or cfg.DATASET.T == 28:
        print('--------------------------------GLU 30 Model-------------------------------')
        model = BiGLUModel(c_in, fw_input_days=cfg.DATASET.T, bw_input_days=cfg.DATASET.T+2).cuda()
    elif cfg.DATASET.T == 21:
        print('--------------------------------GLU 21 Model-------------------------------')
        model = MediumBiGLUModel(c_in, fw_input_days=cfg.DATASET.T, bw_input_days=cfg.DATASET.T+2).cuda()
    elif cfg.DATASET.T == 14:
        print('--------------------------------GLU 21 Model-------------------------------')
        model = ShortBiGLUModel(c_in, fw_input_days=cfg.DATASET.T, bw_input_days=cfg.DATASET.T+2).cuda()

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

    fwflow_x = Variable(torch.FloatTensor(1).cuda())
    bwflow_x = Variable(torch.FloatTensor(1).cuda())
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
        test_iter = iter(test_loader)
        val_len = len(val_iter)
        train_len = len(train_iter)
        # print(val_len, train_len, cfg.LOG.log_time)
        for i in range(cfg.LOG.log_time):
            fwinput_flow_x, bwinput_flow_x, input_flow_y = next(train_iter)
            fwflow_x.resize_(fwinput_flow_x.size()).copy_(fwinput_flow_x)
            bwflow_x.resize_(bwinput_flow_x.size()).copy_(bwinput_flow_x)
            flow_y.resize_(input_flow_y.size()).copy_(input_flow_y)

            preds = model(fwflow_x, bwflow_x)
            loss = criterion(preds, flow_y)

            optimizer.zero_grad()

            loss.backward() 
            optimizer.step()

            train_loss_temp += loss.item()

        for i in range(val_len):
            fwinput_flow_x, bwinput_flow_x, input_flow_y = next(val_iter)
            fwflow_x.resize_(fwinput_flow_x.size()).copy_(fwinput_flow_x)
            bwflow_x.resize_(bwinput_flow_x.size()).copy_(bwinput_flow_x)
            flow_y.resize_(input_flow_y.size()).copy_(input_flow_y)

            preds = model(fwflow_x, bwflow_x)
            loss = criterion(preds, flow_y)

            val_loss_temp += loss.item()

        log.logger.info('Step:{}, Learning_rate:{:.6f}, train_loss:{:.4f}, val_loss:{:.4f}'.format(p, lr, \
                            train_loss_temp/cfg.LOG.log_time, val_loss_temp/(val_len+0.01)))
        train_loss_temp = 0.0
        val_loss_temp = 0.0
        input_t = cfg.DATASET.T + 2
        if (p+1) % cfg.TRAIN.test_time == 0:
            # test
            if date_index == None:
                # middle pred
                fwinput_flow_x, bwinput_flow_x = next(test_iter)
                fwflow_x.resize_(fwinput_flow_x.size()).copy_(fwinput_flow_x)
                bwflow_x.resize_(bwinput_flow_x[:,:,:,:input_t].size()).copy_(bwinput_flow_x[:,:,:,:input_t])
                preds = model(fwflow_x, bwflow_x)

                preds_input = torch.cat([(preds.cpu().double()-mean_flow)/std_flow, fwinput_flow_x[:,3:6,:,cfg.DATASET.T-7:cfg.DATASET.T-6], ], dim=1)# 1 x H x N x 1

                # N x 3 x 1 x 1
                for i in range(1, cfg.DATASET.middle_result_days):
                    bwflow_x.resize_(bwinput_flow_x[:,:,:,i:i+input_t].size()).copy_(bwinput_flow_x[:,:,:,i:i+input_t])
                    # fwinput_flow_x = torch.cat([fwinput_flow_x[:,:3,:,1:], (preds.cpu().double()-mean_flow)/std_flow], dim=-1)
                    fwinput_flow_x = torch.cat([fwinput_flow_x[:,:6,:,1:], preds_input], dim=-1)

                    fwinput_flow_x = fwinput_flow_x.data
                    fwflow_x.resize_(fwinput_flow_x.size()).copy_(fwinput_flow_x)
                    preds = model(fwflow_x, bwflow_x)
                    preds_input = torch.cat([(preds.cpu().double()-mean_flow)/std_flow, fwinput_flow_x[:,3:6,:,cfg.DATASET.T-7:cfg.DATASET.T-6], ], dim=1)

                # generate result
                middle_preds = torch.cat([(fwinput_flow_x[:,:3,:,-1*(cfg.DATASET.middle_result_days-1):] * std_flow) + mean_flow, preds.cpu().double()], dim=-1)# 1 x 3 x N X 15

                np_middle_preds = middle_preds.cpu().data.numpy()
                np_middle_preds = np.transpose(np_middle_preds, axes=[0,2,3,1])

                # break
                middle_date_list, final_date_list = dataset.final_test_date_generate()
                backward_middle_generate_result(np_middle_preds, middle_date_list, p, dataset.district_code_list, dataset.district_city_map())

            # save model on 199
                if p == cfg.TRAIN.epochs-1:
                    torch.save(model.state_dict(), os.path.join(cfg.LOG.experiment_model_dir, 'epoch_199.model'))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.LOG.experiment_model_dir, '{}_epoch_{}.model'.format(date_index,p)))
        model.train()