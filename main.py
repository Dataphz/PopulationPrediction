import os
import argparse 
import numpy as np 
import time 
import torch  
import random 
import torch 

from utils.train_glu import train_GLU
from utils.train_arima import train_auto_arimas
from utils.train_glu_point import train_GLUPoint
from utils.train_glu_fusion import train_GLUFusion
from utils.train_dglu_point import train_DGLUPoint7
from utils.train_glu_point_trend import train_GLUPointTrend
from utils.train_city_glu_point import train_city_models
from utils.train_city_glu_residual_point import train_city_redisual_models
from utils.train_city_glu_residual_point_trend import train_city_redisual_models_trend, train_city_redisual_model_trend
from utils.train_glu_point_tsw import train_GLUPointTrendSeansonWeight
from utils.train_bwglu_point_tsw import train_BWGLUPointTrendSeansonWeight
from utils.train_city_glu_residual_point_tsw import train_city_redisual_model_tsw
from utils.train_biglu_point import train_BiGLUPoint
from utils.finetune_model import finetune
from core.config import PreliminaryConfig as cfg 
from utils.log import log 

if __name__ == '__main__':
    seed = 87#cfg.TRAIN.seed#87#14
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='list of GPU(s) to use', default='0')
    parser.add_argument('--mode', help='mode', default='train')
    parser.add_argument('--model', help='model to train', default='stgcn')
    parser.add_argument('--diff', action='store_true', default=False)
    parser.add_argument('--transform', action='store_true', default=False)
    parser.add_argument('--period', action='store_true', default=False)
    parser.add_argument('--trend', action='store_true', default=False)
    parser.add_argument('--season', action='store_true', default=False)
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--front', action='store_true', default=False)
    parser.add_argument('--final', action='store_true', default=False)
    parser.add_argument('--weighted', action='store_true', default=False)
    parser.add_argument('--predict_days', help='days to predict', default=15)
    parser.add_argument('--t', help='days to input', default=30)
    parser.add_argument('--epochs', help='epochs to train', default=200)
    parser.add_argument('--lr', help='learning rate', default=1e-3)
    parser.add_argument('--lr_schedule', help='learning schedule', default=0)
    parser.add_argument('--test_time', help='period for test', default=10)
    parser.add_argument('--city_index', help='city index for multi-process-predict', default=-1)
    parser.add_argument('--log_time', help='log_time', default=-1)


args = parser.parse_args()
print('--------------------------------------------------------------------')
if args.model == 'arima':
    train_auto_arimas()
elif args.mode == 'train':
    cfg.DATASET.diff = args.diff
    cfg.DATASET.flow_transform = args.transform
    cfg.DATASET.predict_days = int(args.predict_days)
    cfg.DATASET.T = int(args.t)
    cfg.DATASET.trend = args.trend
    cfg.DATASET.season = args.season
    cfg.DATASET.residual = args.residual
    cfg.DATASET.front = args.front 
    cfg.DATASET.final = args.final 
    cfg.DATASET.weighted = args.weighted

    cfg.TRAIN.epochs = int(args.epochs)
    cfg.TRAIN.lr = float(args.lr)
    cfg.TRAIN.period = args.period

    if int(args.log_time) != -1:
        cfg.LOG.log_time = int(args.log_time)

    city_index = int(args.city_index)

    lr_schedule = list()
    print(args.lr_schedule)
    for t in args.lr_schedule.split(','):
        lr_schedule.append(int(t))
    cfg.TRAIN.lr_schedule = lr_schedule
    cfg.TRAIN.test_time = int(args.test_time)

    log.logger.info(cfg)
    print(args)
    torch.utils.backcompat.broadcast_warning.enabled=True

    if args.model == 'glu':
        train_GLU(args)
    elif args.model == 'glu_point':
        if args.season:
            train_GLUPointTrendSeansonWeight(args)
        elif args.residual or args.trend:
            train_GLUPointTrend(args)
        else:
            train_GLUPoint(args)
    elif args.model == 'glu_fusion':
        train_GLUFusion(args)
    elif args.model == 'dglu_7':
        train_DGLUPoint7(args)
    elif args.model == 'residual_city_glu':
        if args.season:
            train_city_redisual_model_tsw(args)
        elif args.trend or args.residual:
            print('train_city_redisual_models_trend')
            if city_index == -1:
                train_city_redisual_models_trend(args)
            else:
                train_city_redisual_model_trend(args, city_index)
        else:
            train_city_redisual_models(args)
    elif args.model == 'city_glu':
        train_city_models(args)
    elif args.model == 'bw_glu_tsw':
        train_BWGLUPointTrendSeansonWeight(args)
    elif args.model == 'biglu_point':
        train_BiGLUPoint(args)

elif args.mode == 'finetune':
    cfg.DATASET.flow_transform = args.transform
    cfg.DATASET.predict_days = int(args.predict_days)
    finetune(args)