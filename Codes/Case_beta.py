import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import argv

DATA_DIR_hf='/home/zheful/risklab/market_beta_new/Results4Xinyu/hf/cbeta'
DATA_DIR_lf='/home/zheful/risklab/market_beta_new/Results4Xinyu/lf/cbeta'
GRAPH_DIR='/project2/dachxiu/xinyu/thesis/Graph_3rd/'
permno =  int(argv[1])
## hf
beta_list_hf = os.listdir(DATA_DIR_hf)
beta_list_hf = list(filter(lambda x: x.endswith('.csv'), beta_list_hf))
beta_paths_hf = list(map(lambda x: os.path.join(DATA_DIR_hf, x), beta_list_hf))
frames = list()
for path in beta_paths_hf:
    if path.find('200506')!=-1:
        continue
    frame = pd.read_csv(path, usecols=['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    frame = frame[frame['permno']==permno]
    frame = frame[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    frame = frame.sort_values(by=['yearmonth', 'permno'])
    frames.append(frame)
beta_df=pd.DataFrame()
beta_df = pd.concat(frames)
beta_df['yearmonth']=pd.to_datetime(beta_df['yearmonth'],format='%Y%m')
beta_df.sort_values(by=['yearmonth'], inplace=True)

columns = ['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']

## lf
beta_list_lf = os.listdir(DATA_DIR_lf)
beta_list_lf = list(filter(lambda x: x.endswith('.csv'), beta_list_lf))
beta_paths_lf = list(map(lambda x: os.path.join(DATA_DIR_lf, x), beta_list_lf))
frames = list()
for path in beta_paths_lf:
    if path.find('200506')!=-1:
        continue
    frame = pd.read_csv(path, usecols=['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    frame = frame[frame['permno']==permno]
    frame = frame[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    frame = frame.sort_values(by=['yearmonth', 'permno'])
    frames.append(frame)
beta_df_lf=pd.DataFrame()
beta_df_lf = pd.concat(frames)
beta_df_lf['yearmonth']=pd.to_datetime(beta_df_lf['yearmonth'],format='%Y%m')
beta_df_lf.sort_values(by=['yearmonth'], inplace=True)

fig =plt.figure(figsize=(8,10))
plt.style.use('seaborn')
for k,factor in enumerate(columns):
    columns = ['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']
    plt.subplot(3, 2, k+1)
    simplex = beta_df[['yearmonth']]
    y = beta_df[[factor]].values
    y_lf = beta_df_lf[[factor]].values
    ax = plt.gca()
    ax.plot_date(simplex, y, linestyle='-', marker='')
#     ax.plot_date(simplex, y_lf, linestyle='-', marker='')
    date_format = mpl.dates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(date_format)
    plt.xlabel('Year')
    fig.autofmt_xdate()
    plt.plot(simplex, y, color='black', label= 'High-frequency beta')
    plt.plot(simplex, y_lf, color='r', linestyle='-.',label= 'Low-frequency beta')
    if k == 0:
        plt.legend()
    plt.title("{}".format(factor), fontsize=16)

if permno == 12490:
    graphname = 'IBM_six_beta.png'
if permno == 12060:
    graphname = 'GE_six_beta.png'
graph_out = os.path.join(GRAPH_DIR, graphname)
plt.savefig(graph_out)
