import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
Out_DIR='D:/UChicago/Thesis/Beta code/Graph/Graph_3rd'
Port = pd.read_csv('monthly_compPort.csv')
Step = pd.read_csv('monthly_compport_step3.csv')
Port = Port[(Port['jdate']>='1998-07-31')]
Port.reset_index(drop = True, inplace=True)
Port.columns = ['Time', 'Stocks in monthly portfolios']
Port['Stocks to calculate high frequency factors and betas']=Step['permno']
Port = Port.set_index('Time')
Port.index = pd.to_datetime(Port.index.values ,format='%Y-%m-%d')
plt.figure(figsize=(8,6))
plt.style.use('seaborn')
Port.plot(kind='line', figsize=(8, 6),colormap='Accent', linewidth=2)
plt.title("Number of stcoks over time in portfolios", fontsize=16)
plt.xlabel('Time')
graphname = "Number of stcoks over time in portfolios"
graph_out = os.path.join(Out_DIR, graphname)
plt.savefig(graph_out)

#####################
# Generate Monthly Stock num in Portfolios
#####################
#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_DIR='/project2/dachxiu/hf_ff_project/Implmt_Code/Xinyu_test/outlier/Portfolio'
Out_DIR='/project2/dachxiu/xinyu/thesis/Graph_3rd'

beta_list = os.listdir(DATA_DIR)
beta_list = list(filter(lambda x: x.endswith('.csv'), beta_list))
beta_paths = list(map(lambda x: os.path.join(DATA_DIR, x), beta_list))
frames = list()
df_needed =pd.DataFrame()
for path in beta_paths:
    datao=pd.read_csv(path,usecols=['permno','jdate','momport','bmport', 'rwport', 'szport', 'caport'])
    data = datao.drop_duplicates()
    df_needed = df_needed.append(data)
df_needed.columns=['permno','jdate','MOM','HML', 'RMW', 'SMB', 'CMA']
df_needed=df_needed.sort_values(by='jdate')
for factor in ['MOM','HML', 'RMW', 'SMB', 'CMA']:
    df=pd.DataFrame(df_needed.groupby('jdate')[factor].value_counts()).rename(columns={factor:'count'}).reset_index().pivot(index='jdate',columns=factor,values='count')
    df.index = pd.to_datetime(df.index.values ,format='%Y-%m-%d')
    compare_df = df
    plt.style.use('seaborn')
    compare_df.plot(kind='line', figsize=(8, 6),colormap='Accent', linewidth=2)
    plt.title("Number of stcoks over time for {}".format(factor), fontsize=16)
    plt.xlabel('Time')
    graphname = 'Number of stcoks in portfolio for {}.png'.format(factor)
    graph_out = os.path.join(Out_DIR, graphname)
    plt.savefig(graph_out)

#####################
# Generate Monthly Stock num in Step3
#####################

#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_DIR='/home/zheful/risklab/market_beta_new/Data'
Out_DIR='/project2/dachxiu/xinyu/thesis/Graph_3rd'

beta_list = os.listdir(DATA_DIR)
beta_list = list(filter(lambda x: x.endswith('.csv') and x.startswith('step3'), beta_list))
beta_paths = list(map(lambda x: os.path.join(DATA_DIR, x), beta_list))
frames = list()
df_needed =pd.DataFrame()
for path in beta_paths:
    data = pd.read_csv(path,usecols =['permno'])
    df_needed[path[-10:-4]] = data[['permno']].nunique()
df_needed = df_needed.T 
df_needed=df_needed.sort_index()
dfname = 'monthly_compport_step3.csv'
df_out = os.path.join(Out_DIR, dfname)
df_needed.to_csv(dfname)
df_needed.to_csv(df_out)
