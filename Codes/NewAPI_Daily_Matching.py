#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from pandas.core.frame import DataFrame
from scipy import stats
import DataHub as hub
import stock

def matchingtable(date):

    startdate = date
    enddate = startdate
    ####
    #Read daily stock price from CRSP
    ####
    CRSP = hub.Handle.create('CRSP')
    crspdaily=CRSP.read('DailyStock', fields='date,permno,prc,ret',permno='%', start=startdate, end=enddate)

    ####
    #Read stock name history from CRSP, to have the tsymbol information
    ####
    crspname=CRSP.read('NameHistory', fields='permno,ticker,shrcls,namedt,nameendt,tsymbol',permno='%')
    crspname=crspname[(crspname.namedt<=startdate) & (crspname.nameendt>=enddate)].sort_values(by=['permno'], ascending=True)
    crspname=crspname[crspname.shrcd.isin([10,11]) & crspname.exchcd.isin([1,2,3])]    
    ####
    #Read TAQ name history from TAQlinktable, to have Yuxing's version of permno-symbol match
    ####
    TAQ = hub.Handle.create('TAQ')
    taqlinktable = TAQ.read('DailyLink', date = 20181231)
    #Read TAQ intraday price, with default permno as 0
    ####
    taq5m = pd.DataFrame(stock.DayTickers(str(date)))
    taq5m.columns = ['symbol','trade_num']
    taq5m.drop('trade_num',axis=1,inplace=True)
    ####
    #Count missing prc from CRSP and save the result
    ####
    result=pd.DataFrame(columns = ["crsp_raw_total","crsp_missing_symbol", "crsp_missing_prc","crsp_negative_prc","missymbol_afterclean_misprc"])
    crsp_missing_prc=crspdaily[crspdaily.prc.isna()]
    crsp_negative_prc=crspdaily[crspdaily.prc<0]
    result.loc[0,['crsp_raw_total']]=crspdaily.date.count()
    result.loc[0,['crsp_missing_prc']]=crsp_missing_prc.date.count()
    result.loc[0,['crsp_negative_prc']]=crsp_negative_prc.date.count()
    #Drop nan prc
    crspdaily=crspdaily.dropna(axis=0, subset=['prc'])
    #Drop negative prc
    crspdaily=crspdaily[crspdaily.prc>0]

    ####
    #Count missing tsymbol from CRSP name and save the result
    ####
    crsp_missing_symbol=crspname[crspname.tsymbol.isna()]
    result.loc[0,['crsp_missing_symbol']]=len(crspname[crspname.tsymbol.isna()])

    ####
    #Merge CRSP stock price with name info
    ####
    crspwithticker=pd.merge(crspdaily,crspname, on="permno", how="inner")
    crspwithticker.rename(columns={'tsymbol':'symbol','ticker':'ticker_crsp'}, inplace=True)
    crsplinktable=crspwithticker[['permno','symbol','ticker_crsp']]
    result.loc[0,['missymbol_afterclean_misprc']]=len(crsplinktable[crsplinktable.symbol.isna()])
    result['crsp_cleaned_total']=len(crspdaily)

    ####
    #fill symbol none using table given by yuxing
    ####
    crspwithtiker_yuxing=pd.merge(crspwithticker,taqlinktable, on="permno", how="left")
    crspwithtiker_yuxing['C'] = crspwithtiker_yuxing.apply(lambda x: (str(x.ticker_crsp) in str(x.ticker))& ((str(x.ticker)[-2:]!='WI')&(str(x.ticker)[-2:]!='WD')&(str(x.ticker)[-1]!='V')), axis=1)
    crspwithtiker_yuxing['CWI'] = crspwithtiker_yuxing.apply(lambda x: (str(x.ticker_crsp) in str(x.ticker))& ((str(x.ticker)[-2:]=='WI')|(str(x.ticker)[-2:]=='WD')|(str(x.ticker)[-1]=='V')), axis=1)
    ###!!! CWI refers to special suffix
    crspwithtiker_yuxing['symbol']=np.where(crspwithtiker_yuxing['symbol'].isna()&(crspwithtiker_yuxing.ticker_crsp==crspwithtiker_yuxing.ticker),crspwithtiker_yuxing['ticker_crsp'],crspwithtiker_yuxing['symbol'])
    crspwithtiker_yuxing['symbol']=np.where(crspwithtiker_yuxing['symbol'].isna()&(crspwithtiker_yuxing.ticker_crsp!=crspwithtiker_yuxing.ticker)&(crspwithtiker_yuxing.CWI==True),crspwithtiker_yuxing['ticker_crsp'],crspwithtiker_yuxing['symbol'])
    crspwithtiker_yuxing['symbol']=np.where(crspwithtiker_yuxing['symbol'].isna()&(crspwithtiker_yuxing.ticker_crsp!=crspwithtiker_yuxing.ticker)&(crspwithtiker_yuxing.C==True),crspwithtiker_yuxing['ticker'],crspwithtiker_yuxing['symbol'])
    crspwithtiker_yuxing['symbol']=np.where(crspwithtiker_yuxing['symbol'].isna(),crspwithtiker_yuxing['ticker_crsp'],crspwithtiker_yuxing['symbol'])
    crspwithtiker_yuxing.drop(columns=['ticker'], inplace=True)
    crsp_missing_symbol_after_yuxing=crspwithtiker_yuxing[crspwithtiker_yuxing.symbol.isna()]
    result['crsp_missing_symbol_after_yuxing']=len(crsp_missing_symbol_after_yuxing)


    ####
    #Count raw total TAQ from taq5m and save the result
    ####
    result['taq_raw_total']=len(taq5m)

    ####
    #Merge cleaned CRSP stock price with taq price on symbol
    ####
    mergeby_tsymbol_symbol=pd.merge(crspwithtiker_yuxing,taq5m,on=['symbol'], how='inner')
    ####
    #Generating result
    ####
    finalmatch_crsp_taq=mergeby_tsymbol_symbol[['permno', 'symbol']]
    #finalmatch_crsp_taq=ignore_nomatch[['permno', 'symbol']]

    return finalmatch_crsp_taq
