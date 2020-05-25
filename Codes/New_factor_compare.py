
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_DIR_PY='/home/zheful/risklab/market_beta_new/Data'
DATA_DIR_lf='/home/zheful/risklab/market_beta_new/Results4Xinyu/lf/cbeta'

GRAPH_DIR='/project2/dachxiu/xinyu/thesis/Graph_3rd'
## hf
beta_list_hf = os.listdir(DATA_DIR_PY)
beta_list_hf = list(filter(lambda x: x.startswith('step4_'), beta_list_hf))
beta_list_hf.sort()
beta_paths_hf = list(map(lambda x: os.path.join(DATA_DIR_PY, x), beta_list_hf))
frames = list()
for path in beta_paths_hf:
#    if path.find('200506')!=-1:
#        continue
    frame = pd.read_csv(path, usecols = ['time'	,'Rm',	'HML'	,'RMW'	,'CMA', 'MOM',	'SMB'])
    frame = frame.sort_values(by=['time'])
    frames.append(frame)
comparison=pd.DataFrame()
comparison = pd.concat(frames)
comparisoncopy = comparison.drop_duplicates(subset=['time'])
comparisoncopy = comparisoncopy.sort_values(by=['time'])
comparisoncopy = comparisoncopy.set_index('time')
comparisoncopy = comparisoncopy+1
comparisoncopy = comparisoncopy.reset_index()
comparisoncopy.time=pd.to_datetime(comparisoncopy.time)

comparisoncopy = comparisoncopy.resample('D', on='time').prod()
comparisoncopy = comparisoncopy[comparisoncopy[comparisoncopy==1].sum(axis=1)!=6]
comparisoncopy.columns = ['MKT',	'HML'	,'RMW'	,'CMA', 'MOM',	'SMB']
comparisoncopy_daily=((comparisoncopy).cumprod()-1)

columns = ['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']
FF_columns = ['FF_MKT', 'FF_HML', 'FF_RMW', 'FF_CMA', 'FF_MOM', 'FF_SMB']
sdate = pd.to_datetime(comparisoncopy_daily.index.values[0]).strftime("%Y-%m-%d")
edate = pd.to_datetime(comparisoncopy_daily.index.values[-1]).strftime("%Y-%m-%d")
import pandas_datareader.data as web  # module for reading datasets directly from the web
#pip install pandas-datareader (in case you haven't install this package)
from pandas_datareader.famafrench import get_available_datasets
Datatoreadff='F-F_Research_Data_5_Factors_2x3_daily'
ds_factorsff = web.DataReader(Datatoreadff,'famafrench',start=sdate,end=edate) # Taking [0] as extracting 1F-F-Research_Data_Factors_2x3')
print('\nKEYS\n{}'.format(ds_factorsff.keys()))
print('DATASET DESCRIPTION \n {}'.format(ds_factorsff['DESCR']))

dfFactorff = ds_factorsff[0].copy()/100
_ff=pd.DataFrame(dfFactorff)
_ff=_ff.reset_index()

_ff=pd.DataFrame(dfFactorff)
_ff.columns=['FF_'+col for col in _ff.columns]
_ff['FF_MKT'] = _ff['FF_RF']+_ff['FF_Mkt-RF']
Datatoread='6_Portfolios_ME_Prior_12_2_Daily'
ds_factors = web.DataReader(Datatoread,'famafrench',start=sdate,end=edate) # Taking [0] as extracting 1F-F-Research_Data_Factors_2x3')
print('\nKEYS\n{}'.format(ds_factors.keys()))
print('DATASET DESCRIPTION \n {}'.format(ds_factors['DESCR']))

dfFactor = ds_factors[0].copy()/100

mom_ff=pd.DataFrame(dfFactor)
mom_ff['FF_MOM']=(mom_ff['BIG HiPRIOR']+mom_ff['SMALL HiPRIOR']-mom_ff['SMALL LoPRIOR']-mom_ff['BIG LoPRIOR'])/2
mom_ff=mom_ff[['FF_MOM']]
# mom_ff.rename(columns = {'Date':'date'}, inplace = True) 
_ff=pd.merge(_ff,mom_ff,left_index=True, right_index=True)
_ff.index = comparisoncopy_daily.index  
_ff=((_ff+1).cumprod()-1)
comparisoncopy_daily=pd.merge(comparisoncopy_daily,_ff,left_index=True, right_index=True)


fig =plt.figure(figsize=(8,10))
plt.style.use('seaborn')
for k,factor in enumerate(columns):
    
    plt.subplot(3, 2, k+1)
    simplex = comparisoncopy_daily.index
    y = comparisoncopy_daily[[factor]].values
    _ff_factor = FF_columns[k]
    ff_y = comparisoncopy_daily[[_ff_factor]].values
#    y_lf = beta_df_lf[[factor]].values 
    ax = plt.gca()
    ax.plot_date(simplex, y, linestyle='-', marker='')
#     ax.plot_date(simplex, y_lf, linestyle='-', marker='')
    date_format = mpl.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(date_format)
    plt.xlabel('Year')
    fig.autofmt_xdate()
    plt.plot(simplex, y, color='r', label= 'High-frequency factor')
    plt.plot(simplex, ff_y, color='b', label= 'Fama-French factor')
#    plt.plot(simplex, y_lf, color='r', linestyle='-.',label= 'Low-frequency beta')
#    if k == 0:
#        plt.legend()
    plt.legend()
    plt.title("{}".format(factor), fontsize=16)
graphname = 'C High Freq Factors.png'
graph_out = os.path.join(GRAPH_DIR, graphname)
plt.savefig(graph_out)

