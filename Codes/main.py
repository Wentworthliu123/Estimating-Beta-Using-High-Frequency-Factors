import numpy as np
import pandas as pd
import pickle
from data_utils import *
from stat_utils import *


def tralpha(year, month, factor_num):
    yrmth = year * 100 + month
    print('Doing', yrmth, '......')

    # load data
    factors, ret_df = getdata(year, month)
    ret_df = ret_df[['date', 'permno', 'retadj', 'me']]
    ret_df.columns = ['date', 'permno', 'return', 'lme']
    date_full = np.unique(ret_df['date'])
    # n_periods = 79 * len(date_full)
    permnolist = np.unique(ret_df['permno'])
    permnonum = len(permnolist)

    # Fill return on missing dates as zero
    fill_ret = lambda ret: fill_miss_date(ret, date_full)
    ret = np.vstack(ret_df.groupby('permno').apply(fill_ret))
    yearmonth = yrmth * np.ones((permnonum, 1))
    ret_df = pd.DataFrame({'ret': ret[:, 2], 'permno': ret[:, 1]})
    factors = factors[:, -factor_num:]

    # Pick lagged market equity for each permno
    # lme = ret[:, 3][0::n_periods]

    hausman_test = lambda ret: hausman_tests_ACn(np.array(ret['ret']), 1 / 12, 3, 0.48, 1)
    hausman_result_ = ret_df.groupby('permno').apply(hausman_test)
    # lf_nonzero_pct = np.array([hausman_result_.iloc[i][0] for i in range(permnonum)])

    # Start estimate
    avgR2_c_mat = np.ones((permnonum, 1))
    JPerc_mat = np.ones((permnonum, 1))
    RV_mat = np.ones((permnonum, 1))
    RV_trunc_mat = np.ones((permnonum, 1))
    ER_c_mat = np.ones((permnonum, 1))
    ER_d_mat = np.ones((permnonum, 1))
    Igamma2_mat = np.ones((permnonum, 6))
    IdJ_mat = np.ones((permnonum, 1))
    IdJ_avar_mat = np.ones((permnonum, 1))
    Ibeta_c_mat = np.ones((permnonum, factor_num * 2))
    Ibeta_j_mat = np.ones((permnonum, factor_num))
    Ibeta_c_avar_mat = np.ones((factor_num, factor_num, permnonum))
    BNS_c_mat = np.ones((permnonum, factor_num))
    BNS_t_mat = np.ones((permnonum, factor_num))
    BNS_c_iv_mat = np.ones((permnonum, 1))
    BNS_t_iv_mat = np.ones((permnonum, 1))
    Igamma2_avar_mat = np.ones((permnonum, 1))
    Xret_all = [1] * permnonum
    Yret_all = [1] * permnonum
    test_freq_all = np.ones((permnonum, 1))

    for permid in range(permnonum):
        test_ret = ret[ret_df['permno'] == permnolist[permid], 2]
        test_freq = 1
        nbseconds = 23400

        test_subret = subsample(test_ret, test_freq)
        subfactor = subsample(factors, test_freq)
        Xret = subfactor.T
        Yret = np.array([test_subret])

        tr_alpha = 5
        tr_beta = 0.47 # 0.49 much better for IdJ estimation. other: 0.47
        HY_flag = 0
        flag_err = 0

        [avgR2_c, JPerc, RV, RV_trunc, ER_c, ER_d, Igamma2, Igamma2_avar, Igamma2_avar_adj, IdJ, IdJ_avar, Ibeta_c, Ibeta_j,
         Ibeta_c_avar, BNS_c, BNS_t, BNS_c_iv, BNS_t_iv] = fn_idio_est_empirical_scatter(Xret, Yret, tr_alpha,nbseconds,
                                                                                         HY_flag, flag_err, test_freq)

        avgR2_c_mat[permid] = avgR2_c
        JPerc_mat[permid] = JPerc
        RV_mat[permid] = RV
        RV_trunc_mat[permid] = RV_trunc
        ER_c_mat[permid] = ER_c
        ER_d_mat[permid] = ER_d
        Igamma2_mat[permid, :] = Igamma2
        IdJ_mat[permid] = IdJ
        IdJ_avar_mat[permid] = IdJ_avar
        Ibeta_c_mat[permid, :] = Ibeta_c
        Ibeta_j_mat[permid, :] = Ibeta_j
        Ibeta_c_avar_mat[:, :, permid] = Ibeta_c_avar
        BNS_c_mat[permid, :] = BNS_c
        BNS_t_mat[permid, :] = BNS_t
        BNS_c_iv_mat[permid] = BNS_c_iv
        BNS_t_iv_mat[permid] = BNS_t_iv
        Igamma2_avar_mat[permid] = Igamma2_avar
        Xret_all[permid] = Xret
        Yret_all[permid] = Yret
        test_freq_all[permid] = test_freq

    # c_beta
    out_df = pd.DataFrame(data=Ibeta_c_mat[:, :factor_num], columns=['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    out_df.to_csv('./Results4Xinyu/lf/cbeta/%d.csv' % yrmth)

    # j_beta
    out_df = pd.DataFrame(data=Ibeta_j_mat, columns=['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    out_df.to_csv('./Results4Xinyu/lf/jbeta/%d.csv' % yrmth)

    # Other results requested by Xinyu
    out_df = pd.DataFrame(data=np.vstack([JPerc_mat.ravel(), RV_mat.ravel(), RV_trunc_mat.ravel(), IdJ_mat.ravel(),
                                          BNS_c_iv_mat.ravel(), BNS_t_iv_mat.ravel()]).T,
                          columns=['JPerc', 'RV', 'RV_trunc', 'IdJ', 'BNS_c', 'BNS_t'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'JPerc', 'RV', 'RV_trunc', 'IdJ', 'BNS_c', 'BNS_t']]
    out_df.to_csv('./Results4Xinyu/lf/other/%d.csv' % yrmth)


def tralpha_hf(year, month, factor_num):
    yrmth = year * 100 + month
    print('Doing', yrmth, '......')

    # load data
    factors, ret_df = getdata(year, month)
    ret_df = ret_df[['date', 'permno', 'retadj', 'me']]
    ret_df.columns = ['date', 'permno', 'return', 'lme']
    date_full = np.unique(ret_df['date'])
    # n_periods = 79 * len(date_full)
    permnolist = np.unique(ret_df['permno'])
    permnonum = len(permnolist)

    # Fill return on missing dates as zero
    fill_ret = lambda ret: fill_miss_date(ret, date_full)
    ret = np.vstack(ret_df.groupby('permno').apply(fill_ret))
    yearmonth = yrmth * np.ones((permnonum, 1))
    ret_df = pd.DataFrame({'ret': ret[:, 2], 'permno': ret[:, 1]})
    factors = factors[:, -factor_num:]

    # Pick lagged market equity for each permno
    # lme = ret[:, 3][0::n_periods]

    # HF BLOCK VVV
    freq = np.array([1, 2, 6, 13]) * 5
    nonzero_thred = 0.9

    # Hausman Test and Non-zero Filter
    ACn_list = np.empty((permnonum, 0))
    nonzero_pct_list = np.empty((permnonum, 0))
    nonzero_num_list = np.empty((permnonum, 0))

    for q in freq:
        hausman_test = lambda ret: hausman_tests_ACn(np.array(ret['ret']), 1 / 12, 3, 0.48, q)
        hausman_result_ = ret_df.groupby('permno').apply(hausman_test)
        nonzero_pct = np.array([hausman_result_.iloc[i][0] for i in range(permnonum)])
        ACn = np.array([hausman_result_.iloc[i][1] for i in range(permnonum)])
        nonzero_num = np.array([hausman_result_.iloc[i][2] for i in range(permnonum)])

        ACn_list = np.hstack((ACn_list, ACn.reshape(permnonum, 1)))
        nonzero_pct_list = np.hstack((nonzero_pct_list, nonzero_pct.reshape(permnonum, 1)))
        nonzero_num_list = np.hstack((nonzero_num_list, nonzero_num.reshape(permnonum, 1)))

    hausman_result = np.hstack(
        (yearmonth, permnolist.reshape(permnonum, 1), nonzero_pct_list, ACn_list, nonzero_num_list))

    # Frequency Selection
    freq_select = abs(hausman_result[:, -2 * len(freq): -len(freq)]) < 2
    nonzero_select = hausman_result[:, 2: 2 + len(freq)] > nonzero_thred
    both_filter = np.multiply(freq_select, nonzero_select)

    freq_list = both_filter * np.append(freq[: -1], 6.5 * 60)
    freq_list[freq_list == 0] = 6.5 * 60
    freq_list = freq_list.min(axis=1)
    freq_list = np.hstack((hausman_result[:, [0, 1]], freq_list.reshape(permnonum, 1)))
    # HF BLOCK ^^^


    hausman_test = lambda ret: hausman_tests_ACn(np.array(ret['ret']), 1 / 12, 3, 0.48, 1)
    hausman_result_ = ret_df.groupby('permno').apply(hausman_test)
    # lf_nonzero_pct = np.array([hausman_result_.iloc[i][0] for i in range(permnonum)])

    # Start estimate
    avgR2_c_mat = np.ones((permnonum, 1))
    JPerc_mat = np.ones((permnonum, 1))
    RV_mat = np.ones((permnonum, 1))
    RV_trunc_mat = np.ones((permnonum, 1))
    ER_c_mat = np.ones((permnonum, 1))
    ER_d_mat = np.ones((permnonum, 1))
    Igamma2_mat = np.ones((permnonum, 6))
    IdJ_mat = np.ones((permnonum, 1))
    IdJ_avar_mat = np.ones((permnonum, 1))
    Ibeta_c_mat = np.ones((permnonum, factor_num * 2))
    Ibeta_j_mat = np.ones((permnonum, factor_num))
    Ibeta_c_avar_mat = np.ones((factor_num, factor_num, permnonum))
    BNS_c_mat = np.ones((permnonum, factor_num))
    BNS_t_mat = np.ones((permnonum, factor_num))
    BNS_c_iv_mat = np.ones((permnonum, 1))
    BNS_t_iv_mat = np.ones((permnonum, 1))
    Igamma2_avar_mat = np.ones((permnonum, 1))
    Xret_all = [1] * permnonum
    Yret_all = [1] * permnonum
    test_freq_all = np.ones((permnonum, 1))

    for permid in range(permnonum):
        print('permid %s' % permid)
        test_ret = ret[ret_df['permno'] == permnolist[permid], 2]

        # HF BLOCK VVV
        test_freq = freq_list[freq_list[:, 1] == permnolist[permid], 2][0]
        nbseconds = test_freq * 60
        # HF BLOCK ^^^

        test_subret = subsample(test_ret, test_freq)
        subfactor = subsample(factors, test_freq)
        Xret = subfactor.T
        Yret = np.array([test_subret])

        tr_alpha = 5
        tr_beta = 0.47 # 0.49 much better for IdJ estimation. other: 0.47
        HY_flag = 0
        flag_err = 0

        [avgR2_c, JPerc, RV, RV_trunc, ER_c, ER_d, Igamma2, Igamma2_avar, Igamma2_avar_adj, IdJ, IdJ_avar, Ibeta_c, Ibeta_j,
         Ibeta_c_avar, BNS_c, BNS_t, BNS_c_iv, BNS_t_iv] = fn_idio_est_empirical_scatter(Xret, Yret, tr_alpha,nbseconds,
                                                                                         HY_flag, flag_err, test_freq)

        avgR2_c_mat[permid] = avgR2_c
        JPerc_mat[permid] = JPerc
        RV_mat[permid] = RV
        RV_trunc_mat[permid] = RV_trunc
        ER_c_mat[permid] = ER_c
        ER_d_mat[permid] = ER_d
        Igamma2_mat[permid, :] = Igamma2
        IdJ_mat[permid] = IdJ
        IdJ_avar_mat[permid] = IdJ_avar
        Ibeta_c_mat[permid, :] = Ibeta_c
        Ibeta_j_mat[permid, :] = Ibeta_j
        Ibeta_c_avar_mat[:, :, permid] = Ibeta_c_avar
        BNS_c_mat[permid, :] = BNS_c
        BNS_t_mat[permid, :] = BNS_t
        BNS_c_iv_mat[permid] = BNS_c_iv
        BNS_t_iv_mat[permid] = BNS_t_iv
        Igamma2_avar_mat[permid] = Igamma2_avar
        Xret_all[permid] = Xret
        Yret_all[permid] = Yret
        test_freq_all[permid] = test_freq

    # c_beta
    out_df = pd.DataFrame(data=Ibeta_c_mat[:, :factor_num], columns=['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    out_df.to_csv('./Results4Xinyu/hf/cbeta/%d.csv' % yrmth)

    # j_beta
    out_df = pd.DataFrame(data=Ibeta_j_mat, columns=['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    out_df.to_csv('./Results4Xinyu/hf/jbeta/%d.csv' % yrmth)

    # Other results requested by Xinyu
    out_df = pd.DataFrame(data=np.vstack([JPerc_mat.ravel(), RV_mat.ravel(), RV_trunc_mat.ravel(), IdJ_mat.ravel(),
                                          BNS_c_iv_mat.ravel(), BNS_t_iv_mat.ravel()]).T,
                          columns=['JPerc', 'RV', 'RV_trunc', 'IdJ', 'BNS_c', 'BNS_t'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'JPerc', 'RV', 'RV_trunc', 'IdJ', 'BNS_c', 'BNS_t']]
    out_df.to_csv('./Results4Xinyu/hf/other/%d.csv' % yrmth)


# Daily beta with +-15d moving window
def tralpha_hf_daily(year, month, day, factor_num):
    yrmth = year * 10000 + month * 100 + day  # should be named yrmthdt but i don't want to change
    print('Doing', yrmth, '......')

    # load data
    factors, ret_df = getdata_anyday(year, month, day)
    ret_df = ret_df[['date', 'permno', 'retadj', 'me']]
    ret_df.columns = ['date', 'permno', 'return', 'lme']
    date_full = np.unique(ret_df['date'])
    # n_periods = 79 * len(date_full)
    permnolist = np.unique(ret_df['permno'])
    permnonum = len(permnolist)

    # Fill return on missing dates as zero
    fill_ret = lambda ret: fill_miss_date(ret, date_full)
    ret = np.vstack(ret_df.groupby('permno').apply(fill_ret))
    yearmonth = yrmth * np.ones((permnonum, 1))
    ret_df = pd.DataFrame({'ret': ret[:, 2], 'permno': ret[:, 1]})
    factors = factors[:, -factor_num:]

    # Pick lagged market equity for each permno
    # lme = ret[:, 3][0::n_periods]

    # HF BLOCK VVV
    freq = np.array([1, 2, 6, 13]) * 5
    nonzero_thred = 0.9

    # Hausman Test and Non-zero Filter
    ACn_list = np.empty((permnonum, 0))
    nonzero_pct_list = np.empty((permnonum, 0))
    nonzero_num_list = np.empty((permnonum, 0))

    for q in freq:
        hausman_test = lambda ret: hausman_tests_ACn(np.array(ret['ret']), 1 / 12, 3, 0.48, q)
        hausman_result_ = ret_df.groupby('permno').apply(hausman_test)
        nonzero_pct = np.array([hausman_result_.iloc[i][0] for i in range(permnonum)])
        ACn = np.array([hausman_result_.iloc[i][1] for i in range(permnonum)])
        nonzero_num = np.array([hausman_result_.iloc[i][2] for i in range(permnonum)])

        ACn_list = np.hstack((ACn_list, ACn.reshape(permnonum, 1)))
        nonzero_pct_list = np.hstack((nonzero_pct_list, nonzero_pct.reshape(permnonum, 1)))
        nonzero_num_list = np.hstack((nonzero_num_list, nonzero_num.reshape(permnonum, 1)))

    hausman_result = np.hstack(
        (yearmonth, permnolist.reshape(permnonum, 1), nonzero_pct_list, ACn_list, nonzero_num_list))

    # Frequency Selection
    freq_select = abs(hausman_result[:, -2 * len(freq): -len(freq)]) < 2
    nonzero_select = hausman_result[:, 2: 2 + len(freq)] > nonzero_thred
    both_filter = np.multiply(freq_select, nonzero_select)

    freq_list = both_filter * np.append(freq[: -1], 6.5 * 60)
    freq_list[freq_list == 0] = 6.5 * 60
    freq_list = freq_list.min(axis=1)
    freq_list = np.hstack((hausman_result[:, [0, 1]], freq_list.reshape(permnonum, 1)))
    # HF BLOCK ^^^

    # Start estimate
    avgR2_c_mat = np.ones((permnonum, 1))
    JPerc_mat = np.ones((permnonum, 1))
    RV_mat = np.ones((permnonum, 1))
    RV_trunc_mat = np.ones((permnonum, 1))
    ER_c_mat = np.ones((permnonum, 1))
    ER_d_mat = np.ones((permnonum, 1))
    Igamma2_mat = np.ones((permnonum, 6))
    IdJ_mat = np.ones((permnonum, 1))
    IdJ_avar_mat = np.ones((permnonum, 1))
    Ibeta_c_mat = np.ones((permnonum, factor_num * 2))
    Ibeta_j_mat = np.ones((permnonum, factor_num))
    Ibeta_c_avar_mat = np.ones((factor_num, factor_num, permnonum))
    BNS_c_mat = np.ones((permnonum, factor_num))
    BNS_t_mat = np.ones((permnonum, factor_num))
    BNS_c_iv_mat = np.ones((permnonum, 1))
    BNS_t_iv_mat = np.ones((permnonum, 1))
    Igamma2_avar_mat = np.ones((permnonum, 1))
    Xret_all = [1] * permnonum
    Yret_all = [1] * permnonum
    test_freq_all = np.ones((permnonum, 1))

    for permid in range(permnonum):
        test_ret = ret[ret_df['permno'] == permnolist[permid], 2]

        # HF BLOCK VVV
        test_freq = freq_list[freq_list[:, 1] == permnolist[permid], 2][0]
        nbseconds = test_freq * 60
        # HF BLOCK ^^^

        test_subret = subsample(test_ret, test_freq)
        subfactor = subsample(factors, test_freq)
        Xret = subfactor.T
        Yret = np.array([test_subret])

        tr_alpha = 5
        tr_beta = 0.47 # 0.49 much better for IdJ estimation. other: 0.47
        HY_flag = 0
        flag_err = 0

        [avgR2_c, JPerc, RV, RV_trunc, ER_c, ER_d, Igamma2, Igamma2_avar, Igamma2_avar_adj, IdJ, IdJ_avar, Ibeta_c, Ibeta_j,
         Ibeta_c_avar, BNS_c, BNS_t, BNS_c_iv, BNS_t_iv] = fn_idio_est_empirical_scatter(Xret, Yret, tr_alpha,nbseconds,
                                                                                         HY_flag, flag_err, test_freq)

        avgR2_c_mat[permid] = avgR2_c
        JPerc_mat[permid] = JPerc
        RV_mat[permid] = RV
        RV_trunc_mat[permid] = RV_trunc
        ER_c_mat[permid] = ER_c
        ER_d_mat[permid] = ER_d
        Igamma2_mat[permid, :] = Igamma2
        IdJ_mat[permid] = IdJ
        IdJ_avar_mat[permid] = IdJ_avar
        Ibeta_c_mat[permid, :] = Ibeta_c
        Ibeta_j_mat[permid, :] = Ibeta_j
        Ibeta_c_avar_mat[:, :, permid] = Ibeta_c_avar
        BNS_c_mat[permid, :] = BNS_c
        BNS_t_mat[permid, :] = BNS_t
        BNS_c_iv_mat[permid] = BNS_c_iv
        BNS_t_iv_mat[permid] = BNS_t_iv
        Igamma2_avar_mat[permid] = Igamma2_avar
        Xret_all[permid] = Xret
        Yret_all[permid] = Yret
        test_freq_all[permid] = test_freq
    out_df = pd.DataFrame(data=Ibeta_c_mat[:, :factor_num], columns=['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    out_df.to_csv('./Result_hf_daily/beta_%d.csv' % yrmth)


# Daily beta with data that day (5min frequency only)
def tralpha_hf_daily_local(year, month, day, factor_num):
    yrmth = year * 10000 + month * 100 + day  # should be named yrmthdt but i don't want to change
    print('Doing', yrmth, '......')

    # load data
    factors, ret_df = getdata_anyday_local(year, month, day)
    ret_df = ret_df[['date', 'permno', 'retadj', 'me']]
    ret_df.columns = ['date', 'permno', 'return', 'lme']
    date_full = np.unique(ret_df['date'])
    permnolist = np.unique(ret_df['permno'])
    permnonum = len(permnolist)

    # Fill return on missing dates as zero
    fill_ret = lambda ret: fill_miss_date(ret, date_full)
    ret = np.vstack(ret_df.groupby('permno').apply(fill_ret))
    yearmonth = yrmth * np.ones((permnonum, 1))
    ret_df = pd.DataFrame({'ret': ret[:, 2], 'permno': ret[:, 1]})
    factors = factors[:, -factor_num:]

    # Start estimate
    avgR2_c_mat = np.ones((permnonum, 1))
    JPerc_mat = np.ones((permnonum, 1))
    RV_mat = np.ones((permnonum, 1))
    RV_trunc_mat = np.ones((permnonum, 1))
    ER_c_mat = np.ones((permnonum, 1))
    ER_d_mat = np.ones((permnonum, 1))
    Igamma2_mat = np.ones((permnonum, 6))
    IdJ_mat = np.ones((permnonum, 1))
    IdJ_avar_mat = np.ones((permnonum, 1))
    Ibeta_c_mat = np.ones((permnonum, factor_num * 2))
    Ibeta_j_mat = np.ones((permnonum, factor_num))
    Ibeta_c_avar_mat = np.ones((factor_num, factor_num, permnonum))
    BNS_c_mat = np.ones((permnonum, factor_num))
    BNS_t_mat = np.ones((permnonum, factor_num))
    BNS_c_iv_mat = np.ones((permnonum, 1))
    BNS_t_iv_mat = np.ones((permnonum, 1))
    Igamma2_avar_mat = np.ones((permnonum, 1))
    Xret_all = [1] * permnonum
    Yret_all = [1] * permnonum

    for permid in range(permnonum):
        test_ret = ret[ret_df['permno'] == permnolist[permid], 2]

        # HF BLOCK VVV
        test_freq = 5
        nbseconds = test_freq * 60
        # HF BLOCK ^^^

        test_subret = subsample(test_ret, test_freq)
        subfactor = subsample(factors, test_freq)
        Xret = subfactor.T
        Yret = np.array([test_subret])

        tr_alpha = 5
        tr_beta = 0.47  # 0.49 much better for IdJ estimation. other: 0.47
        HY_flag = 0
        flag_err = 0

        [avgR2_c, JPerc, RV, RV_trunc, ER_c, ER_d, Igamma2, Igamma2_avar, Igamma2_avar_adj, IdJ, IdJ_avar, Ibeta_c, Ibeta_j,
         Ibeta_c_avar, BNS_c, BNS_t, BNS_c_iv, BNS_t_iv] = fn_idio_est_empirical_scatter(Xret, Yret, tr_alpha,nbseconds,
                                                                                         HY_flag, flag_err, test_freq)

        avgR2_c_mat[permid] = avgR2_c
        JPerc_mat[permid] = JPerc
        RV_mat[permid] = RV
        RV_trunc_mat[permid] = RV_trunc
        ER_c_mat[permid] = ER_c
        ER_d_mat[permid] = ER_d
        Igamma2_mat[permid, :] = Igamma2
        IdJ_mat[permid] = IdJ
        IdJ_avar_mat[permid] = IdJ_avar
        Ibeta_c_mat[permid, :] = Ibeta_c
        Ibeta_j_mat[permid, :] = Ibeta_j
        Ibeta_c_avar_mat[:, :, permid] = Ibeta_c_avar
        BNS_c_mat[permid, :] = BNS_c
        BNS_t_mat[permid, :] = BNS_t
        BNS_c_iv_mat[permid] = BNS_c_iv
        BNS_t_iv_mat[permid] = BNS_t_iv
        Igamma2_avar_mat[permid] = Igamma2_avar
        Xret_all[permid] = Xret
        Yret_all[permid] = Yret
    out_df = pd.DataFrame(data=Ibeta_c_mat[:, :factor_num], columns=['MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB'])
    out_df['permno'] = permnolist
    out_df['yearmonth'] = yearmonth.ravel()
    out_df = out_df[['yearmonth', 'permno', 'MKT', 'HML', 'RMW', 'CMA', 'MOM', 'SMB']]
    out_df.to_csv('./Result_hf_daily_local/beta_%d.csv' % yrmth)

if __name__ == '__main__':
    # tralpha(2000, 1, 6)
    tralpha_hf(2000, 1, 6)
