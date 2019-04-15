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
from utils.train_glu_point import train_GLUPoint


def train_DGLUPoint7(args):

    # for i in range(7):
    #     print('------------training model {}------------'.format(i))
    #     train_GLUPoint(args, date_index=i) # generate model_i

    dataset = DataGenerator(cfg.FILE.flow_file, cfg.FILE.transfer_file, cfg.FILE.process_transition_file, t_period=cfg.DATASET.T, 
                n_days=cfg.DATASET.days, predict_days=cfg.DATASET.predict_days, nodes=cfg.DATASET.nodes, train_proportion=cfg.DATASET.train_proportion)
    train_transition, train_flow_x, train_flow_y, val_transition, val_flow_x, val_flow_y, test_transition, test_flow_x = dataset.generate()
    test_dataset = TestDatasetLoader(test_transition, test_flow_x)
    test_loader = DataLoader(dataset = test_dataset, batch_size=1, num_workers=0, shuffle=False)
    # generate test_file

    # first 7 days
    flow_x = torch.FloatTensor(1).cuda()
    transition_x = torch.FloatTensor(1).cuda()
    flow_y = torch.FloatTensor(1).cuda()

    mean_flow = torch.DoubleTensor(np.transpose(dataset.mean_flow, axes=[0,3,1,2]))
    std_flow = torch.DoubleTensor(np.transpose(dataset.std_flow, axes=[0,3,1,2]))


    preds_tensor = torch.DoubleTensor(0,0,0,0)
    for w in range(3):
        preds7 = list()
        for i in range(7):
            print(w,i)
            model = GLUModel(c_in=3, input_days=cfg.DATASET.T).cuda()
            model.load_state_dict(torch.load(os.path.join(cfg.LOG.home_log_dir, '12-13-14_DGLU7_Train/model', '{}_epoch_199.model'.format(i))))
            model.eval()
            test_iter = iter(test_loader)
            input_transition_x, input_flow_x = next(test_iter)
            if w > 0:
                input_flow_x = torch.cat((input_flow_x[:,:,:,7*w:], preds_tensor[:,:,:,:w*7]), dim=-1) # 1 x 3 x N x 30
            input_flow_x = input_flow_x.data

            transition_x.resize_(input_transition_x.size()).copy_(input_transition_x)
            flow_x.resize_(input_flow_x.size()).copy_(input_flow_x)
            preds = model(flow_x, transition_x, constant_attention=True) # 1 x 3 x N X 1
            preds7.append(preds)
        preds7_tensor = torch.cat(preds7, dim=-1).cpu().double() # 1 x 3 x N x 7
        preds7_tensor = (preds7_tensor - mean_flow) / std_flow
        preds_tensor = torch.cat((preds_tensor, preds7_tensor), dim=-1) # 1 x 3 x N x 7*w

    # 1 x N x 15 x 3
    preds_tensor = (preds_tensor * std_flow) + mean_flow
    print(preds_tensor.shape)
    np_preds = preds_tensor.cpu().data.numpy()
    np_preds = np.transpose(np_preds[:,:,:,:15], axes=[0,2,3,1])
    generate_result(np_preds, 199, dataset.test_date_generate(), dataset.district_code_list, dataset.district_city_map())

