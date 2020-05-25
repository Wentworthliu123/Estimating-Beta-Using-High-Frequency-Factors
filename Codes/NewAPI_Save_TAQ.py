#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import DataHub as hub
import datetime
import NewAPI_Daily_Matching
import os.path
import stock

portfoliopath = '/project2/dachxiu/hf_ff_project/Implmt_Code/Xinyu_test/outlier/Portfolio'
outlierpath='/project2/dachxiu/hf_ff_project/Implmt_Code/Xinyu_test/outlier/Outlier'
taqoutpath='/project2/dachxiu/hf_ff_project/Implmt_Code/Xinyu_test/outlier/TAQdata'
matchoutpath='/project2/dachxiu/hf_ff_project/Implmt_Code/Xinyu_test/outlier/TAQmatch'
outpath='/project2/dachxiu/hf_ff_project/Implmt_Code/Xinyu_test/outlier/Output'

######################
# Time Range Setting #
######################
startyear = 2018
endyear = 2020

    
for year in range(startyear, endyear):

    CSV_FILE_PATH = f'{str(year)}0701_0630_daily_all_RCC.csv'
    portfolio_to_read = os.path.join(portfoliopath, CSV_FILE_PATH)
    pdata = pd.read_csv(portfolio_to_read)
    pdata['date']=pd.to_datetime(pdata['date'])
    datecomplete = list(map(lambda x: x.strftime("%Y%m%d"),pdata.date))
    datelist=list(set(datecomplete))
    datelist = [x for x in datelist if x>='20190101']
    datelist.sort()
    
#    for eachday in datelist:
#        kd = NewAPI_Daily_Matching.matchingtable(int(eachday))
#        kd['date']=pd.to_datetime(eachday)
#        # Save daily matchtable to the disc
#        matchtable_to_save = os.path.join(matchoutpath, "matchtable_"+str(eachday)+".csv")
#        kd.to_csv(matchtable_to_save, index=False)

    for eachday in datelist:
        df = pd.DataFrame(stock.DayTickers(str(eachday)))
        df.columns = ['symbol','trade_num']
        df.drop('trade_num',axis=1,inplace=True)
        TAQ_df = pd.DataFrame(index = df['symbol'], columns = np.arange(79))
        for stock_symbol in df['symbol']:
            price5min = pd.DataFrame(stock.Query5Min(str(eachday), stock_symbol))
            TAQ_df.loc[stock_symbol] = price5min.iloc[1,:]
        TAQ_df.reset_index(inplace = True)
        dff=pd.DataFrame()
        dff = pd.melt(TAQ_df, id_vars=list(TAQ_df.columns)[:1], value_vars=list(TAQ_df.columns)[1:], \
                      var_name='intratime', value_name='tprice')
        dff.columns = ['symbol', 'intratime','tprice']
        dff = dff.sort_values(by=['symbol', 'intratime'])
        dff.reset_index(drop='true')
        dff['date']=pd.to_datetime(eachday)
        taq_to_save = os.path.join(taqoutpath, "taq_"+str(eachday)+".csv")
        dff.to_csv(taq_to_save, index=False)