#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import os.path

#DATA_DIR = '/project2/dachxiu/xinyu/thesis/DATA'
DATA_DIR = '/home/zheful/risklab/market_beta_new/Data'
RESULT_DIR = '/project2/dachxiu/xinyu/thesis/RESULT'
TEMP_DIR = '/project2/dachxiu/xinyu/thesis/TEMP'

#######################################################################################################################
#                                                BUILD STEP3                                                          #
#######################################################################################################################

step3_list = os.listdir(TEMP_DIR)
step3_list = list(filter(lambda x: x.endswith('.csv') and x.startswith('pre_step3_'), step3_list))
step3_list.sort()
step3_paths = list(map(lambda x: os.path.join(TEMP_DIR, x), step3_list))
frames = list()
for path in step3_paths:
    frame = pd.read_csv(path, usecols=['date', 'permno', 'retadj', 'wt', 'me'])
    frame = frame[['date', 'permno', 'retadj', 'wt', 'me']]
    frame = frame.sort_values(by=['date', 'permno'])
    frames.append(frame)
step3_df = pd.concat(frames)
step3_df['ym'] = step3_df['date'].apply(lambda x: int(''.join(x[:7].split('-'))))
step3_group = step3_df.groupby('ym')
for group in list(step3_group.groups.keys()):
    out_df = step3_group.get_group(group)
    out_df = out_df.sort_values(by=['date', 'permno'])
    del out_df['ym']
    out_df.to_csv(os.path.join(DATA_DIR, 'step3_%d.csv' % group), index=False)

    
#######################################################################################################################
#                                                BUILD STEP4                                                          #
#######################################################################################################################

step4_list = os.listdir(TEMP_DIR)
step4_list = list(filter(lambda x: x.endswith('.csv') and x.startswith('pre_step4_'), step4_list))
step4_list.sort()
step4_paths = list(map(lambda x: os.path.join(TEMP_DIR, x), step4_list))
frames = list()
for path in step4_paths:
    frames.append(pd.read_csv(path))
step4_df = pd.concat(frames)
step4_df['ym'] = step4_df['time'].apply(lambda x: int(''.join(x[:7].split('-'))))
step4_group = step4_df.groupby('ym')
for group in list(step4_group.groups.keys()):
    out_df = step4_group.get_group(group)
    out_df = out_df.sort_values(by=['time'])
    del out_df['ym']
    out_df.to_csv(os.path.join(DATA_DIR, 'step4_%d.csv' % group), index=False)