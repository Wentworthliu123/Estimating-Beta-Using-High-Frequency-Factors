import numpy as np
import pandas as pd
from data_utils import *


def fn_spot_cov(rret1, kn, T, tr_alpha, tr_beta):
    bv = sum(abs(np.multiply(rret1[1:], rret1[:-1]))) * np.pi / 2 / T
    Jidx = abs(rret1) < (tr_alpha * (T / len(rret1)) ** tr_beta * np.sqrt(bv))
    chat = np.zeros(len(rret1) - kn + 1)
    for j in range(len(rret1) - kn + 1):
        chat[j] = 1 / kn / (T / len(rret1)) * sum(np.multiply(rret1[j: j + kn] ** 2, Jidx[j: j + kn]))
    return chat


def hausman_tests_ACn(ret, T, tr_alpha, tr_beta, freq):
    ret = subsample(ret, freq)
    rret = ret[ret != 0]

    n = len(rret)
    if n == 0:
        return 0, np.nan, 0

    threshold = tr_alpha * (T / n) ** tr_beta * np.sqrt(sum(abs(np.multiply(rret[1:], rret[:-1]))) * np.pi / 2 / T)
    Jidx = abs(rret) > threshold
    quarticity_hat_cont = sum(np.multiply(rret ** 4, Jidx)) / T / 3 * n

    kn = int(np.floor(np.sqrt(n)))
    chat = fn_spot_cov(rret, kn, T, tr_alpha, tr_beta)
    quarticity_hat_jump = sum(np.multiply(np.multiply(1 - Jidx[kn: n - kn], ret[kn: n - kn] ** 2),
                                          chat[: n - 2 * kn] + chat[kn: n - kn]))
    quarticity_hat = quarticity_hat_cont + quarticity_hat_jump

    ACn = np.sqrt(1 / T) * sum(np.multiply(rret[1:], rret[:-1])) / np.sqrt(quarticity_hat) * np.sqrt(n)
    nonzero_pct = sum(ret != 0) / len(ret)
    nonzero_num = sum(ret != 0)

    return nonzero_pct, ACn, nonzero_num


def fn_idx_1variable(tr_alpha, tr_beta, Xret, deltan, ndays, nobs):
    trunc_const = np.multiply(tr_alpha, deltan ** tr_beta)
    XretMat = Xret.reshape(nobs, ndays, order='F')
    bv = (np.multiply(abs(XretMat[1:, :]), abs(XretMat[:-1, :])) * np.pi / 2 * nobs / (nobs - 1)).sum(axis=0) * 252
    bv = np.kron(bv, np.ones((nobs, 1))).reshape(nobs * ndays, 1, order='F')
    bv = bv.astype(float)
    Xidx = (abs(Xret) < trunc_const * np.sqrt(bv).T) * 1
    return Xidx


def fn_idx(tr_alpha, tr_beta, Xret, deltan, ndays, nobs):
    if Xret.shape[0] == 1:
        Xidx = fn_idx_1variable(tr_alpha, tr_beta, Xret, deltan, ndays, nobs)
    else:
        Xidx = -999 * np.ones(Xret.shape)
        p = Xret.shape[0]
        for ix_p in range(p):
            Xidx[ix_p, :] = fn_idx_1variable(tr_alpha, tr_beta, Xret[ix_p, :], deltan, ndays, nobs)
    return Xidx


def rcond(A):
    return 1 / np.linalg.norm(A, 1) / np.linalg.norm(np.linalg.inv(A), 1)


def fn_idio_est_empirical_scatter(Xret, Yret, tr_alpha, nbseconds, HY_flag, flag_err, test_freq):
    Xret = np.nan_to_num(Xret)
    Yret = np.nan_to_num(Yret)

    n = Xret.shape[1]
    p = Xret.shape[0]
    deltan = 1 / 252 / (60 * 60 * 6.5) * nbseconds
    T = 1 / 12

    if nbseconds == 300:
        kn = 79
        n78 = 79
        ndays = int(n / n78)
    elif nbseconds == 300 * 2:
        kn = 40
        n78 = 40
        ndays = int(n / n78)
    elif nbseconds == 300 * 6:
        kn = 14
        n78 = 14
        ndays = int(n / n78)
    elif nbseconds == 300 * 13:
        kn = 7
        n78 = 7
        ndays = int(n / n78)
    elif nbseconds == 300 * 12 * 6.5:
        kn = n
        n78 = n
        ndays = 1
    else:
        print('bad number of seconds')
        return None

    if test_freq == 1:
        ndays = 1
        kn = n
        n78 = n

    theta = kn * np.sqrt(deltan)

    if HY_flag == 2:
        deltan = T / sum(Yret != 0)
        ix = Yret == 0
        P_x = np.vstack((np.zeros((1, p)), np.cumsum(Xret.T, axis=0)))
        Xret = np.diff(P_x[np.append([True], Yret != 0), :], axis=0).T
        Yret = Yret[Yret != 0]
        n = Xret.shape[1]
        HY_flag = 0

        ndays = 1
        n78 = n
        if nbseconds == 300:
            kn = 78
        elif nbseconds == 300 * 2:
            kn = 78
        elif nbseconds == 300 * 6:
            kn = 39
        elif nbseconds == 300 * 13:
            kn = 30
        elif nbseconds == 300 * 12 * 6.5:
            kn = n
        else:
            print('bad number of seconds')
            return None

    if (kn > n) or (Xret.shape[0] > Xret.shape[1]) or flag_err or len(Yret) < 1:
        avgR2_c = np.nan
        JPerc = np.nan
        RV = np.nan
        RV_trunc = np.nan
        ER_c = np.nan
        ER_d = np.nan
        Igamma2 = np.nan * np.ones((1, 3))
        IdJ = np.nan
        Ibeta_c = np.nan * np.ones((1, 2 * p))
        Ibeta_j = np.nan * np.ones((1, p))
        Ibeta_c_avar = np.nan * np.ones((p, p))
        BNS_c = np.nan * np.ones(p)
        BNS_t = np.nan * np.ones(p)
        BNS_c_iv = np.nan
        BNS_t_iv = np.nan
        Igamma2_avar = np.nan
        Igamma2_avar_adj = np.nan
        IdJ_avar = np.nan
        return avgR2_c, JPerc, RV, RV_trunc, ER_c, ER_d, Igamma2, Igamma2_avar, Igamma2_avar_adj, IdJ, IdJ_avar, Ibeta_c, Ibeta_j, Ibeta_c_avar, BNS_c, BNS_t, BNS_c_iv, BNS_t_iv

    tr_beta = 0.47
    Xidx = fn_idx(tr_alpha, tr_beta, Xret, deltan, ndays, n78)
    Yidx = fn_idx(tr_alpha, tr_beta, Yret, deltan, ndays, n78)

    # the continuous part
    Xret_c = np.multiply(Xret, Xidx)
    Yret_c = np.multiply(Yret, Yidx)
    ER_c = Yret_c.sum()

    if HY_flag:
        # TODO: fn_spot_beta_HY needed
        # BNS_c = fn_spot_beta_HY(Xret_c.T, Yret_c.T)
        # BNS_t = fn_spot_beta_HY(Xret.T, Yret.T)
        pass
    else:
        # print('=====')
        # print(Xret_c.astype(str))
        # print('=====')
        # print(Yret_c.T.astype(str))
        # print('=====')
        # print(Xret.astype(str))
        # print('=====')
        # print(Yret.astype(str))

        BNS_c = np.linalg.lstsq(np.dot(Xret_c, Xret_c.T).astype(float), np.dot(Xret_c, Yret_c.T).astype(float))[0].reshape(p)  # continuous
        BNS_t = np.linalg.lstsq(np.dot(Xret, Xret.T).astype(float), np.dot(Xret, Yret.T).astype(float))[0].reshape(p)  # ols (total)

    # realized vol of y (SST)
    RV_trunc = (Yret_c ** 2).sum() / T
    RV = (Yret ** 2).sum() / T

    # SSR
    BNS_c_iv = ((Yret_c - np.dot(BNS_c.T, Xret_c)) ** 2).sum()
    BNS_t_iv = ((Yret - np.dot(BNS_t.T, Xret)) ** 2).sum()

    # the jump part
    Xret_d = np.multiply(Xret, 1 - Xidx)
    Yret_d = np.multiply(Yret, 1 - Yidx)[0]
    ER_d = Yret_d.sum()

    # percentage of jumps
    if (Yret ** 2).sum() > 0:
        JPerc = (Yret_d ** 2).sum() / (Yret ** 2).sum()
    else:
        JPerc = 0

    nJ_X = Xidx[0, :].size - Xidx[0, :].sum()
    nJ_Y = Yidx.size - Yidx.sum()
    del Xret, Xidx, Yidx

    # We only calculate the jump beta if exactly one factor jumps(o/w, no ID)
    ix_1jump = np.sum(Xret_d != 0, axis=0) == 1
    Ibeta_j = np.nan * np.ones(p)
    if len(ix_1jump) > 0:
        for i in range(p):
            Ibeta_spot = np.nan * np.ones(n)
            ix = (Xret_d[i, :] != 0) & ix_1jump
            Ibeta_spot[ix] = np.divide(Yret_d[ix], Xret_d[i, ix])
            Ibeta_j[i] = (Ibeta_spot[~np.isnan(Ibeta_spot)]).mean()

    Yret_c = np.append(Yret_c, np.zeros(kn))

    spot_Cy = np.zeros(int(n / kn))
    spot_beta = np.zeros((p, int(n / kn)))
    spot_ibav = np.zeros((p, p, int(n / kn)))
    spot_g2_p1 = np.zeros(int(n / kn))
    spot_g2_p2 = np.zeros(int(n / kn))
    spot_g2 = np.zeros(int(n / kn))

    for idx in range(0, n, kn):
        j = int(idx / kn)
        Xj = Xret_c[:, idx: idx + kn].T
        Yj = Yret_c[idx: idx + kn].T

        XpX = np.dot(Xj.T, Xj)
        c = XpX / kn / deltan

        b = np.nan * np.ones((p, 1))
        if rcond(XpX) > 10e-15:
            if HY_flag:
                # need fn_spot_beta_HY
                # b = fn_spot_beta_HY(Xj,Yj);
                pass
            else:
                b = np.linalg.lstsq(XpX.astype(float), np.dot(Xj.T, Yj).astype(float))[0]

        if np.any(np.isnan(b)):
            if j > 0:
                b = spot_beta[:, j - 1]
            else:
                # b = np.zeros((p, 1))
                b = np.zeros(p)

        spot_Cy[j] = np.dot(Yj.T, Yj) / kn / deltan
        spot_beta[:, j] = b
        spot_g2_p1[j] = np.dot(Yj.T, Yj) / kn / deltan
        spot_g2_p2[j] = np.dot(np.dot(b.T, c), b)
        spot_g2[j] = spot_g2_p1[j] - spot_g2_p2[j]
        spot_ibav[:, :, j] = np.nan * np.ones(c.shape)
        if rcond(XpX) > 10e-15:
            spot_ibav[:, :, j] = spot_g2[j] * np.linalg.inv(c)

    spot_Cy = np.kron(spot_Cy.reshape(int(n / kn), 1), np.ones((kn, 1))).T

    # IdJ is sum(Yret_d.^2), but only for times when none of X jumps.
    IdJ = (np.multiply(Yret_d ** 2, np.sum(Xret_d == 0, axis=0) == p)).sum() / T
    IdJ_avar = 4 * (np.multiply(Yret_d ** 2, spot_Cy)).sum() * deltan / (T ** 2)

    Ibeta_c = np.sum(spot_beta * deltan * kn, 1).T / T
    A1 = 0
    Ibeta_c = np.append(Ibeta_c, Ibeta_c + A1)
    Ibeta_c_avar = np.sum(spot_ibav, 2) * deltan * kn * deltan / (T ** 2)

    Igamma2_p1 = (1 + p / kn) * np.sum(spot_g2_p1 * deltan * kn) / T
    Igamma2_p2 = (1 + p / kn) * np.sum(spot_g2_p2 * deltan * kn) / T

    A1_p1 = 0
    A1_p2 = 0
    scale1 = n / (n - nJ_Y)
    scale2 = n / (n - nJ_X)

    est1 = Igamma2_p1 - Igamma2_p2
    est2 = (Igamma2_p1 + A1_p1) - (Igamma2_p2 + A1_p2)
    est3 = (Igamma2_p1 + A1_p1) * scale1 - (Igamma2_p2 + A1_p2) * scale2
    est4 = est1 * n / n
    est5 = (Igamma2_p1 + A1_p1 * (1 + p / kn)) - (Igamma2_p2 + A1_p2 * (1 + p / kn))
    est_naive = spot_g2.sum() * deltan * kn
    Igamma2 = [est1, est2, est3, est4, est5, est_naive]
    Igamma2_avar = 2 * ((spot_g2_p1 - spot_g2_p2) ** 2).sum() * deltan * kn * deltan / (T ** 2)
    # Igamma2_avar_adj = kn * deltan * (spot_g2[0] ** 2 + spot_g2[-1] ** 2)
    Igamma2_avar_adj = kn * deltan * (spot_g2 ** 2).mean()

    K = int(np.floor(n / kn))
    spot_r2 = np.zeros((K, 1))
    for j in range(K):
        Xj_c = Xret_c[:, j * kn: (j + 1) * kn].T
        Yj_c = Yret_c[j * kn: (j + 1) * kn].T
        b = spot_beta[:, j]

        Y_hat_c = np.dot(Xj_c, b)
        spot_r2[j] = (Y_hat_c ** 2).sum() / (Yj_c ** 2).sum()
        if np.isnan(spot_r2[j]):
            spot_r2[j] = 0
    avgR2_c = np.mean(spot_r2)

    return avgR2_c, JPerc, RV, RV_trunc, ER_c, ER_d, Igamma2, Igamma2_avar, Igamma2_avar_adj, IdJ, IdJ_avar, Ibeta_c, Ibeta_j, Ibeta_c_avar, BNS_c, BNS_t, BNS_c_iv, BNS_t_iv
