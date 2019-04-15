import os
import socket 
import time 

from core.collections import AttrDict
from utils.log import log 
from utils.common_utils import checkdir

name = socket.gethostname()
print('host_name:', name)
HOME_PATH_MAP = {'slave2':'/disk5/penghao','dataology-desktop':'/disk4/penghao', 'fantasyvm8':'/data/penghao', 'gpgpu002':'/home/xxc'}
HOME_PATH = os.path.join(HOME_PATH_MAP[name], 'workspace/competition/population_prediction_JD/population_prediction/dataset/data')

PreliminaryConfig = AttrDict()
_C = PreliminaryConfig

_C.HOME_PATH = HOME_PATH

# Model Selection & Description
_C.MODEL.NAME = 'GLUPointCityResidualTrend'
_C.MODEL.DESCRIPTION = 'Train_Smoothing_Drop_holiday_front'#Augment82_ALLDataset'#'Train'#+DateFeaturewithoutNormal'

# File Setting
_C.FILE.front_flow_file = os.path.join(HOME_PATH, 'front_smoothing_drop_holiday_flow_train.csv')
_C.FILE.final_flow_file = os.path.join(HOME_PATH, 'final_smoothing_flow_train.csv')
# _C.FILE.final_flow_file = os.path.join(HOME_PATH, 'augment_final_smoothing_flow_train.csv')
# _C.FILE.final_flow_file = os.path.join(HOME_PATH, 'final_smoothing_drop_holiday_flow_for_final_prediction_train.csv')
# _C.FILE.final_flow_file = os.path.join(HOME_PATH, 'final_smoothing_drop_holiday_flow_for__prediction_train2.csv')
# _C.FILE.complete_flow_file = os.path.join(HOME_PATH, 'augment_complete_smoothing_flow_train.csv')# [88,92]不能取，是预测区间
_C.FILE.complete_flow_file = os.path.join(HOME_PATH, 'augment_complete_smoothing_flow_train_refinefinalformiddle.csv')# [88,92]不能取，是预测区间
_C.FILE.front_transfer_file = os.path.join(HOME_PATH, 'front_transition_train.csv')
_C.FILE.final_transfer_file = os.path.join(HOME_PATH, 'final_transition_train.csv')
_C.FILE.process_front_transition_file = os.path.join(HOME_PATH, 'front_transition_data.pkl')
_C.FILE.process_final_transition_file = os.path.join(HOME_PATH, 'final_transition_data.pkl')

# Dataset Setting
_C.DATASET.T = 30
_C.DATASET.result_days = 15
_C.DATASET.middle_result_days = 5
_C.DATASET.final_result_days = 10
_C.DATASET.predict_days = 1
_C.DATASET.days = 101#274
_C.DATASET.nodes = 204
_C.DATASET.train_proportion = 0.8#
_C.DATASET.flow_transform = False
_C.DATASET.diff = False
_C.DATASET.trend = False
_C.DATASET.season = False
_C.DATASET.residual = False
_C.DATASET.front = True
_C.DATASET.final = True
_C.DATASET.weighted = False


# Train Setting
_C.TRAIN.seed = 87#1130# 
_C.TRAIN.epochs = 200#400
_C.TRAIN.bs = 256
_C.TRAIN.lr = 1e-3
_C.TRAIN.weight_decay = 1e-5
_C.TRAIN.lr_schedule = [100]#[20,50,100]#
_C.TRAIN.test_time = 10
_C.TRAIN.period = False

# Finetune Setting
_C.FINETUNE.lr = 1e-5
_C.FINETUNE.epochs = 100
_C.FINETUNE.lr_schedule = []
_C.FINETUNE.test_time = 1
_C.FINETUNE.log_time = 46

# Log Setting
logtime = time.strftime("%m-%d-%H", time.localtime())
_C.LOG.home_log_dir = checkdir(os.path.join(HOME_PATH_MAP[name], 'workspace/competition/population_prediction_JD/log'))
_C.LOG.timestamp = '{}_{}_{}'.format(logtime, _C.MODEL.NAME, _C.MODEL.DESCRIPTION)
_C.LOG.experiment_log_dir = checkdir(os.path.join(_C.LOG.home_log_dir, _C.LOG.timestamp))
_C.LOG.experiment_result_dir = checkdir(os.path.join(_C.LOG.experiment_log_dir, 'result'))
_C.LOG.experiment_model_dir = checkdir(os.path.join(_C.LOG.experiment_log_dir, 'model'))

# 21:70 30:63
_C.LOG.log_time =  63#100#82#70#  80
log.initialize_log(os.path.join(_C.LOG.experiment_log_dir, 'record.log'))
log.logger.info(logtime)




