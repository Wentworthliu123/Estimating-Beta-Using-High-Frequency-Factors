import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sys import argv

DATA_DIR='/project2/dachxiu/zheful/market_beta_new/Results4Xinyu/hf/other/'
GRAPH_DIR='/project2/dachxiu/xinyu/thesis/Graph_3rd/'
permno = int(argv[1])
beta_list = os.listdir(DATA_DIR)
beta_list = list(filter(lambda x: x.endswith('.csv'), beta_list))
beta_paths = list(map(lambda x: os.path.join(DATA_DIR, x), beta_list))
frames = list()
for path in beta_paths:
    if path.find('200506')!=-1:
        continue
    frame = pd.read_csv(path, index_col=0)
    frame = frame.sort_values(by=['yearmonth', 'permno'])
    frames.append(frame)
beta_df=pd.DataFrame()
beta_df = pd.concat(frames)
beta_df['yearmonth']=pd.to_datetime(beta_df['yearmonth'],format='%Y%m')
beta_df.reset_index(drop=True, inplace=True)


columns1 = ['RV','RV_trunc', 'BNS_t','BNS_c','JPerc']
columns2 = ['JPerc']
Case_df = beta_df[beta_df['permno']==permno][['yearmonth']+columns1].set_index('yearmonth')

Case_df[['BNS_t','BNS_c']]=Case_df[['BNS_t','BNS_c']]*12
Case_df['Jumps'] = Case_df['JPerc']*Case_df['BNS_c']
plt.style.use('seaborn')
Case_df0 = Case_df[['RV', 'BNS_c' ,'Jumps']]
Case_df0.columns = ['Total risk', 'Total idiosyncratic risk', 'Jump risk']
Case_df0.sort_index(inplace=True)
Case_df0.plot(color = ['#333333', '#BB0000', '#0000BB'], linewidth = 2)

Case_df0['zero'] = 0
plt.fill_between(Case_df0.index, Case_df0['zero'], Case_df0.iloc[:,0] , color='gray', alpha = 0.15)
plt.fill_between(Case_df0.index, Case_df0['zero'],  Case_df0.iloc[:,1]  , color='red', alpha = 0.15)
plt.fill_between(Case_df0.index, Case_df0['zero'],  Case_df0.iloc[:,2]  , color='blue', alpha = 0.15)
graphname = 'GE_{}.png'.format(columns1[0])
graph_out = os.path.join(GRAPH_DIR, graphname)
plt.xlabel('Year')
plt.ylabel('Variation')
plt.savefig(graph_out)


Jperc_df = beta_df[beta_df['permno']==permno][['yearmonth']+columns2].set_index('yearmonth')
Jperc_df['R2'] = (Case_df['RV']-Case_df['BNS_t'])/Case_df['RV']
Jperc_df.columns = ['Jump in idiosyncrastic risk', 'Systematic risk in total risk (R2)']
Jperc_df.plot(color = ['#0000BB', '#BB0000'], linewidth = 2, figsize=(8,3))
plt.xlabel('Year')
plt.ylabel('Fraction')
graphname = 'GE_{}.png'.format(columns2[0])
graph_out = os.path.join(GRAPH_DIR, graphname)
plt.savefig(graph_out)