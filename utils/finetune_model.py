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

def finetune(args):
    seed = cfg.TRAIN.seed#87
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print('Dataset Loading...')
    st = time.clock()
    dataset = DataGenerator(cfg.FILE.flow_file, cfg.FILE.transfer_file, cfg.FILE.process_transition_file, t_period=cfg.DATASET.T, 
                n_days=cfg.DATASET.days, predict_days=cfg.DATASET.predict_days, nodes=cfg.DATASET.nodes, train_proportion=cfg.DATASET.train_proportion, )
    train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, test_flow_x = dataset.generate()

    # train_dataset = TrainDatasetLoader(train_transition, train_flow_x, train_flow_y)
    # train_loader = DataLoader(dataset = train_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    val_dataset = TrainDatasetLoader(val_transition, val_flow_x, val_flow_y)
    val_loader = DataLoader(dataset = val_dataset, batch_size=cfg.TRAIN.bs, num_workers=0, shuffle=True)
    test_dataset = TestDatasetLoader(test_transition, test_flow_x)
    test_loader = DataLoader(dataset = test_dataset, batch_size=1, num_workers=0, shuffle=False)
    print('Dataset Loaded, using {}'.format(time.clock()-st))

    mean_flow = torch.DoubleTensor(np.transpose(dataset.mean_flow, axes=[0,3,1,2]))
    std_flow = torch.DoubleTensor(np.transpose(dataset.std_flow, axes=[0,3,1,2]))

    c_in = train_flow_x.shape[-1]
    model = GLUModel(c_in).cuda()
    model.load_state_dict(torch.load(os.path.join(cfg.LOG.home_log_dir, '12-12-14_GLUPoint_Train/model', 'epoch_199.model')))#1214
    params = []

    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value], 'lr':cfg.FINETUNE.lr,\
                'weight_decay': 0}]
            else:
                params += [{'params':[value], 'lr':cfg.FINETUNE.lr,\
                'weight_decay': cfg.TRAIN.weight_decay}]

    criterion = RMSLELoss().cuda()
    optimizer = torch.optim.SGD(params, momentum=0.9)

    flow_x = Variable(torch.FloatTensor(1).cuda())
    transition_x = Variable(torch.FloatTensor(1).cuda())
    flow_y = Variable(torch.FloatTensor(1).cuda())

    finetune_loss_temp = 0.0

    lr = cfg.FINETUNE.lr
    model.train()

    for p in range(cfg.FINETUNE.epochs):

        if p in cfg.FINETUNE.lr_schedule:
            print('changing learning rate.....')
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * param_group['lr']
                lr = param_group['lr']

        # train 
        val_iter = iter(val_loader)
        test_iter = iter(test_loader)

        val_len = len(val_iter)
        test_len = len(test_iter)

        for i in range(cfg.FINETUNE.log_time//5):
            input_transition_x, input_flow_x, input_flow_y = next(val_iter)
            transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
            flow_y.resize_(input_flow_y.size()).copy_(input_flow_y)

            preds = model(flow_x, transition_x, constant_attention=True)
            loss = criterion(preds, flow_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            finetune_loss_temp += loss.item()

        log.logger.info('Step:{}, Learning_rate:{}, finetune_loss:{:.4f}'.format(p, lr, \
                                finetune_loss_temp/(cfg.FINETUNE.log_time//5+0.01))) 
        finetune_loss_temp = 0.0

        model.eval()
        if (p+1) % cfg.FINETUNE.test_time == 0:
            # test
            input_transition_x, input_flow_x = next(test_iter)
            transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
            preds = model(flow_x, transition_x, constant_attention=True) # 1 x 3 x N X 1
            
            for i in range(1, cfg.DATASET.result_days):
                input_flow_x = torch.cat([input_flow_x[:,:,:,1:], (preds.cpu().double()-mean_flow)/std_flow], dim=-1)
                input_flow_x = input_flow_x.data
                flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
                preds = model(flow_x, transition_x, constant_attention=True)
            # generate result
            
            preds = torch.cat([(input_flow_x[:,:,:,-1*(cfg.DATASET.result_days-1):] * std_flow) + mean_flow, preds.cpu().double()], dim=-1)# 1 x 3 x N X 15
        
            np_preds = preds.cpu().data.numpy()
            np_preds = np.transpose(np_preds, axes=[0,2,3,1])
            # break
            generate_result(np_preds, p, dataset.test_date_generate(), dataset.district_code_list, dataset.district_city_map())

        model.train()