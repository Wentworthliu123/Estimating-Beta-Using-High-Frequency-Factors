import numpy as np
import pandas as pd
from scipy.io import loadmat
import datetime


def agg_quantile(dat, qntl):
    return np.quantile(dat, qntl, axis=0)


def fill_miss_date(ret, date_full):
    adj_ret = np.empty((0, 4))
    ret = np.array(ret)
    permno_ = ret[0, 1]
    if ret.shape[0] < 79 * len(date_full):
        for date_id in range(len(date_full)):
            subdata = ret[ret[:, 0] == date_full[date_id]]
            if len(subdata) < 1:
                insert_data = np.array([[date_full[date_id], permno_, 0, 0] for i in range(79)])
            elif len(subdata) > 79:
                insert_data = subdata[:79]
            else:
                insert_data = subdata
            adj_ret = np.vstack((adj_ret, insert_data))
    elif ret.shape[0] > 79 * len(date_full):
        for date_id in range(len(date_full)):
            subdata = ret[ret[:, 0] == date_full[date_id]]
            if len(subdata) < 1:  # not sure this case should be kept for robustness
                insert_data = np.array([[date_full[date_id], permno_, 0, 0] for i in range(79)])
            elif len(subdata) > 79:
                insert_data = subdata[:79]
            else:
                insert_data = subdata
            adj_ret = np.vstack((adj_ret, insert_data))
    else:
        adj_ret = ret
    return adj_ret


def subsample(ret, freq):
    d = ret.ndim
    if d == 1:
        ret = ret.reshape(len(ret), 1)

    all_adj_ret = np.empty((0, ret.shape[1]))

    if freq == 1 or freq == 390:
        for k in range(0, len(ret), 79):
            subret = ret[k: 79 + k, :]
            subret_cum = np.prod(subret + 1, axis=0) - 1
            all_adj_ret = np.vstack((all_adj_ret, subret_cum))
    else:
        for k in range(0, len(ret), 79):
            seg = int(freq / 5)
            subret = ret[k: 79 + k, :]
            adj_ret = np.empty((0, ret.shape[1]))
            try:
                for i in range(1, 79, seg):
                    subret_cum = np.prod(subret[i: i + seg, :] + 1, axis=0) - 1
                    adj_ret = np.vstack((adj_ret, subret_cum))
            except:
                return
            subret = np.vstack((subret[0, :], adj_ret))
            all_adj_ret = np.vstack((all_adj_ret, subret))

    if d == 1:
        all_adj_ret = all_adj_ret.reshape(len(all_adj_ret), )
    return all_adj_ret


def getdata(year, month):
    # TODO: specify the process to read the data
    #
    # deprecated input codes for MATLAB files (.mat)
    # halfday = loadmat("./Data/HalfTradingDays2020.mat")
    # half_trading_days = halfday['HalfTradingDays']
    #
    # VVVVV
    # dat3 = h5py.File("./Data/step3_" + str(year * 100 + month) + ".mat")
    # dat4 = h5py.File("./Data/step4_" + str(year * 100 + month) + ".mat")
    #
    # step3 = np.array(dat3['step3']).T
    # step3 = step3[~np.isin(step3[:, 0].astype(np.int), half_trading_days), :]
    # dat3 = pd.DataFrame(step3[:, [0, 1, 4, 8]],
    #                     columns=['date', 'permno', 'return', 'lme'])  # date, permno id, return, lme
    #
    # factors = np.array(dat4['return_matrix']).T  # date, time, hml, smb, rmw, cma, mom, mkt
    # factors = factors[~np.isin(factors[:, 0].astype(np.int), half_trading_days), :]
    # ^^^^^

    # read in data from .csv files
    halfday = loadmat("HalfTradingDays2020.mat")
    half_trading_days = halfday['HalfTradingDays']
    half_trading_days = list(map(lambda x: x[0], half_trading_days))
    dat3 = pd.read_csv('./Data/step3_' + str(year * 100 + month) + '.csv')
    columns = dat3.columns
    dat3 = dat3.groupby(['date', 'permno'])
    temp = list()
    for g in dat3:
        vals = g[1].values[:79]
        subdf = pd.DataFrame(data=vals, columns=columns)
        temp.append(subdf)
    dat3 = pd.concat(temp)
    dat3 = dat3.sort_values(by=['date', 'permno'])

    dat4 = pd.read_csv('./Data/step4_' + str(year * 100 + month) + '.csv')
    dat4 = dat4.drop_duplicates(subset=['time'])
    dat4 = dat4.sort_values(by=['time'])

    dat3['date'] = dat3['date'].apply(lambda x: int(''.join(x.split('-'))))

    dates = dat4['time'].apply(lambda x: int(''.join(x[:10].split('-'))))
    times = dat4['time'].apply(lambda x: int(''.join(x[11:].split(':'))))
    factors = np.hstack(
        (dates.values.reshape(len(dates), 1), times.values.reshape(len(dates), 1), dat4.iloc[:, 1:].values))

    dat3 = dat3[~np.isin(dat3['date'].astype(np.int), half_trading_days)]
    factors = factors[~np.isin(factors[:, 0].astype(np.int), half_trading_days), :]

    return factors, dat3


def getdailydata(year, month):
    factors, ret_df = getdata(year, month)

    factors[:, 2:] += 1
    factors_df = pd.DataFrame(np.delete(factors, 1, axis=1))
    daily_factors = factors_df.groupby(0).prod()
    daily_factors -= 1

    ret_df['return'] += 1
    ret_df = ret_df.drop('lme', axis=1)
    daily_ret = ret_df.groupby(['date', 'permno']).prod().reset_index()
    daily_ret['return'] -= 1

    return daily_factors, daily_ret


# plus-minus 15 days
def getdata_anyday(year, month, day, chunksize=50000):
    date0 = datetime.datetime.strptime('%s-%s-%s' % (year, month, day), '%Y-%m-%d')
    range_date = [(date0 + datetime.timedelta(_)).strftime('%Y-%m-%d') for _ in range(-15, 16)]

    path_0 = str(year * 100 + month)
    path_p1 = str(year * 100 + month + 1) if month != 12 else str((year + 1) * 100 + 1)
    path_m1 = str(year * 100 + month - 1) if month != 1 else str((year - 1) * 100 + 12)

    dat3_l = list()
    dat4_l = list()

    for path in [path_m1, path_0, path_p1]:
        try:
            for chunk in pd.read_csv('./Data/step3_' + path + '.csv', chunksize=chunksize):
                chunk_dt = chunk['date']
                chunk = chunk[np.isin(chunk_dt, range_date)]
                dat3_l.append(chunk)
            for chunk in pd.read_csv('./Data/step4_' + path + '.csv', chunksize=chunksize):
                chunk_dt = list(map(lambda x: x[:10], chunk['time']))
                chunk = chunk[np.isin(chunk_dt, range_date)]
                dat4_l.append(chunk)
        except FileNotFoundError:
            continue
    dat3 = pd.concat(dat3_l)
    dat4 = pd.concat(dat4_l)
    if dat3.size == 0 or dat4.size == 0:
        raise LookupError('[Error] No corresponding csv files found. Input DataFrame empty.\n'
                          '(This is a custom error.)')

    halfday = loadmat("HalfTradingDays2020.mat")
    half_trading_days = halfday['HalfTradingDays']
    half_trading_days = list(map(lambda x: x[0], half_trading_days))
    columns = dat3.columns
    dat3 = dat3.groupby(['date', 'permno'])
    temp = list()
    for g in dat3:
        vals = g[1].values[:79]
        subdf = pd.DataFrame(data=vals, columns=columns)
        temp.append(subdf)
    dat3 = pd.concat(temp)
    dat3 = dat3.sort_values(by=['date', 'permno'])

    dat4 = dat4.drop_duplicates(subset=['time'])
    dat4 = dat4.sort_values(by=['time'])

    dat3['date'] = dat3['date'].apply(lambda x: int(''.join(x.split('-'))))

    dates = dat4['time'].apply(lambda x: int(''.join(x[:10].split('-'))))
    times = dat4['time'].apply(lambda x: int(''.join(x[11:].split(':'))))
    factors = np.hstack(
        (dates.values.reshape(len(dates), 1), times.values.reshape(len(dates), 1), dat4.iloc[:, 1:].values))

    dat3 = dat3[~np.isin(dat3['date'].astype(np.int), half_trading_days)]
    factors = factors[~np.isin(factors[:, 0].astype(np.int), half_trading_days), :]

    return factors, dat3


# one day data
def getdata_anyday_local(year, month, day):
    # read in data from .csv files
    path = str(year * 100 + month)

    dat3 = pd.read_csv('./Data/step3_' + path + '.csv')
    dat4 = pd.read_csv('./Data/step4_' + path + '.csv')

    date0 = datetime.datetime.strptime('%s-%s-%s' % (year, month, day), '%Y-%m-%d')
    range_date = [date0.strftime('%Y-%m-%d')]
    dat3_dt = dat3['date']
    dat3 = dat3[np.isin(dat3_dt, range_date)]
    dat4_dt = list(map(lambda x: x[:10], dat4['time']))
    dat4 = dat4[np.isin(dat4_dt, range_date)]

    dat3['date'] = dat3['date'].apply(lambda x: int(''.join(x.split('-'))))
    dates = dat4['time'].apply(lambda x: int(''.join(x[:10].split('-'))))
    times = dat4['time'].apply(lambda x: int(''.join(x[11:].split(':'))))

    factors = np.hstack((
        dates.values.reshape(len(dates), 1), times.values.reshape(len(dates), 1), dat4.iloc[:, 1:].values
    ))

    return factors, dat3
