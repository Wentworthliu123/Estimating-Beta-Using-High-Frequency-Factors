#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_DIR_hf='/home/zheful/risklab/market_beta_new/Results4Xinyu/hf/cbeta'
DATA_DIR_lf='/home/zheful/risklab/market_beta_new/Results4Xinyu/lf/cbeta'

GRAPH_DIR='/project2/dachxiu/xinyu/thesis/Graph_3rd'
## hf
beta_list_hf = os.listdir(DATA_DIR_hf)
beta_list_hf = list(filter(lambda x: x.endswith('.csv'), beta_list_hf))
beta_list_hf.sort()
beta_paths_hf = list(map(lambda x: os.path.join(DATA_DIR_hf, x), beta_list_hf))
frames = list()
for path in beta_paths_hf:
    if path.find('200506')!=-1:
        continue
    frame = pd.read_csv(path, usecols=['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    frame = frame[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    frame = frame.sort_values(by=['yearmonth', 'permno'])
    frames.append(frame)
beta_df=pd.DataFrame()
beta_df = pd.concat(frames)
beta_df['yearmonth']=pd.to_datetime(beta_df['yearmonth'],format='%Y%m')

columns = ['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']

## lf
beta_list_lf = os.listdir(DATA_DIR_lf)
beta_list_lf = list(filter(lambda x: x.endswith('.csv'), beta_list_lf))
beta_list_lf.sort()
beta_paths_lf = list(map(lambda x: os.path.join(DATA_DIR_lf, x), beta_list_lf))
frames = list()
for path in beta_paths_lf:
    if path.find('200506')!=-1:
        continue
    frame = pd.read_csv(path, usecols=['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    frame = frame[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    frame = frame.sort_values(by=['yearmonth', 'permno'])
    frames.append(frame)
beta_df_lf=pd.DataFrame()
beta_df_lf = pd.concat(frames)
beta_df_lf['yearmonth']=pd.to_datetime(beta_df_lf['yearmonth'],format='%Y%m')


for factor in  columns:
    columns = ['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']
    fig =plt.figure(figsize=(6,8))
    plt.style.use('seaborn')
    
    plt.subplot(2, 1, 1)
    lower = beta_df.groupby('yearmonth').quantile(0.25)[factor]
    upper = beta_df.groupby('yearmonth').quantile(0.75)[factor]
    simplex = beta_df.groupby('yearmonth').median().index.values
    y = beta_df.groupby('yearmonth').median()[factor].values
    

    ax = plt.gca()
    ax.plot_date(simplex, y, linestyle='-', marker='')
    date_format = mpl.dates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(date_format)
    plt.xlabel(factor)
    #更改日期格式
    fig.autofmt_xdate()
    plt.plot(simplex, y, color='r')
    plt.fill_between(simplex, lower, upper , color='r', alpha = 0.15)
    plt.ylim(top = 3, bottom =-2)
    plt.title("High(red) v.s. daily(black) for {}".format(factor), fontsize=16)

    ## second plot
    plt.subplot(2, 1, 2)
    lower = beta_df_lf.groupby('yearmonth').quantile(0.25)[factor]
    upper = beta_df_lf.groupby('yearmonth').quantile(0.75)[factor]
    simplex = beta_df_lf.groupby('yearmonth').median().index.values
    y = beta_df_lf.groupby('yearmonth').median()[factor].values
    

    ax = plt.gca()
    ax.plot_date(simplex, y, linestyle='-', marker='')
    date_format = mpl.dates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(date_format)
    plt.xlabel(factor)
    
    #更改日期格式
    fig.autofmt_xdate()
    plt.plot(simplex, y, color='black')
    plt.fill_between(simplex, lower, upper , color='black', alpha = 0.15)
    plt.ylim(top = 3, bottom =-2)
    graphname = '{}.png'.format(factor)
    graph_out = os.path.join(GRAPH_DIR, graphname)
    plt.savefig(graph_out)
