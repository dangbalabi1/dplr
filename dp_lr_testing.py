#!/usr/bin/env python3

import numpy as np
import math
import sys
import optparse
from numpy.linalg import inv

################################ Utilities ################################

def normx(barx, varx, n):
    return np.random.normal(barx, math.sqrt(varx), n)

def unifx(low, high, n):
    return np.random.uniform(low, high, n)

def expx(scale, n):
    return np.random.exponential(scale,  n)

def ranks(ls):
    sortedls = sorted(ls)
    return [sortedls.index(x)+1 for x in ls]

def to_clip(mat, Delta):
    dim = mat.reshape(-1, 1).shape[1]
    if dim == 1:
        return np.clip(mat, 0, Delta)
    else:
        return mat*np.clip(Delta/np.linalg.norm(mat, axis=-1)).reshape(-1, 1)

################################ Test Statistics ################################

# non-DP test statistic for testing linear relationship
def np_lin(x, y, n):
    # means
    xm = np.mean(x)
    ym = np.mean(y)

    # means of degree two
    x2m = np.mean(x**2)
    xym = np.mean(x*y)
    y2m = np.mean(y**2)

    # covariance, variance, slope, and intercept
    cov = (xym - xm*ym)
    var = (x2m - xm**2)
    b1 = cov/var
    b2 = ym - b1*xm

    # standard errors
    s2_0 = (n*y2m + n*b2**2 - 2*n*ym*b2)/(n-2)
    s2_num = (n*y2m + n*b2**2 - 2*n*ym*b2 - 2*n*xym*b1 + 2*n*xm*b1*b2 + n*x2m*b1**2)
    s2 = s2_num/(n-2)

    # test statistic
    f_num = (n-2)*b1**2*n*var
    f_den = s2_num
    f_stat = f_num/f_den

    return (xm, x2m, b1, b2, s2_0, s2, f_stat)

# DP test statistic for testing linear relationship
def lin(x, y, n, rho, Delta):
    rho = float(rho)/5

    # means
    cxm = np.mean(np.clip(x, -Delta, Delta))
    pxm = cxm + np.random.normal(0, math.sqrt(2*Delta**2/(rho*n**2)))
    cym = np.mean(np.clip(y, -Delta, Delta))
    pym = cym + np.random.normal(0, math.sqrt(2*Delta**2/(rho*n**2)))

    # means of degree two
    cx2m = np.mean(np.clip((x)**2, 0, Delta**2))
    px2m = cx2m + np.random.normal(0, math.sqrt(Delta**4/(2*rho*n**2)))
    cxym = np.mean(np.clip((x)*(y), -Delta**2, Delta**2))
    pxym = cxym + np.random.normal(0, math.sqrt(2*Delta**4/(rho*n**2)))
    cy2m = np.mean(np.clip((y)**2, 0, Delta**2))
    py2m = cy2m + np.random.normal(0, math.sqrt(Delta**4/(2*rho*n**2)))

    # covariance, variance, slope, and intercept
    pcov = (pxym - pxm*pym)
    pvar = (px2m - pxm**2)
    pb1 = pcov/pvar
    pb2 = pym - pb1*pxm

    # standard errors
    ps2_0 = (n*py2m + n*pb2**2 - 2*n*pym*pb2)/(n-2)
    ps2_num = (n*py2m + n*pb2**2 - 2*n*pym*pb2 - 2*n*pxym*pb1 + 2*n*pxm*pb1*pb2 + n*px2m*pb1**2)
    ps2 = ps2_num/(n-2)

    # test statistic
    pf_num = (n-2)*pb1**2*n*pvar
    pf_den = ps2_num
    pf_stat = pf_num/pf_den

    return (pxm, px2m, pb1, pb2, ps2_0, ps2, pf_stat)

# non-DP test statistic for testing mixtures
def np_mix(x, y, n1, n):
    n2 = n-n1
    # partition data 
    x1 = x[0:n1]
    x2 = x[n1:]
    y1 = y[0:n1]
    y2 = y[n1:]

    # means
    xm1 = np.mean(x1)
    xm2 = np.mean(x2)
    ym1 = np.mean(y1)
    ym2 = np.mean(y2)
    xm = float(n1)/n * xm1 + float(n2)/n * xm2

    # means of degree two
    x2m1 = np.mean((x1)**2)
    x2m2 = np.mean((x2)**2)
    x2m = float(n1)/n * x2m1 + float(n2)/n * x2m2
    xym1 = np.mean(x1*y1)
    xym2 = np.mean(x2*y2)
    xym = float(n1)/n * xym1 + float(n2)/n * xym2
    y2m1 = np.mean((y1)**2)
    y2m2 = np.mean((y2)**2)
    y2m = float(n1)/n * y2m1 + float(n2)/n * y2m2

    # slopes
    b1 = xym1/x2m1
    b2 = xym2/x2m2

    # standard errors
    s2_0 = (n*y2m1 + n*(b1)**2*x2m1 - 2*n*xym1*b1)/(n-2)
    s2 = (n1*y2m1 + n1*(b1)**2*x2m1 - 2*n1*xym1*b1 + n2*y2m2 + n*(b2)**2*x2m2 - 2*n2*xym2*b2)/(n-2)

    # test statistic
    f_stat = (n1*x2m1*n2*x2m2)*(b1-b2)**2/(s2*n*x2m)

    return (xm1, xm2, xm, x2m1, x2m2, x2m, b1, b2, s2_0, s2, f_stat)

# DP test statistic for testing mixtures
def mix(x, y, n1, n, rho, Delta):
    rho = float(rho)/8
    n2 = n-n1

    # partition data
    x1 = x[0:n1]
    x2 = x[n1:]
    y1 = y[0:n1]
    y2 = y[n1:]

    # means
    cxm1 = np.mean(np.clip(x1, -Delta, Delta))
    pxm1 = cxm1 + np.random.normal(0, math.sqrt(2*Delta**2/(rho*n1**2)))
    cxm2 = np.mean(np.clip(x2, -Delta, Delta))
    pxm2 = cxm2 + np.random.normal(0, math.sqrt(2*Delta**2/(rho*n2**2)))
    pxm = float(n1)/n * pxm1 + float(n2)/n * pxm2

    # means of degree two
    cx2m1 = np.mean(np.clip((x1)**2, 0, Delta**2))
    px2m1 = cx2m1 + np.random.normal(0, math.sqrt(Delta**4/(2*rho*n1**2)))
    cx2m2 = np.mean(np.clip((x2)**2, 0, Delta**2))
    px2m2 = cx2m2 + np.random.normal(0, math.sqrt(Delta**4/(2*rho*n2**2)))
    px2m = float(n1)/n * px2m1 + float(n2)/n * px2m2
    cxym1 = np.mean(np.clip((x1)*(y1), -Delta**2, Delta**2))
    pxym1 = cxym1 + np.random.normal(0, math.sqrt(2*Delta**4/(rho*n1**2)))
    cxym2 = np.mean(np.clip((x2)*(y2), -Delta**2, Delta**2))
    pxym2 = cxym2 + np.random.normal(0, math.sqrt(2*Delta**4/(rho*n2**2)))
    pxym = float(n1)/n * pxym1 + float(n2)/n * pxym2
    cy2m1 = np.mean(np.clip((y1)**2, 0, Delta**2))
    py2m1 = cy2m1 + np.random.normal(0, math.sqrt(Delta**4/(2*rho*n1**2)))
    cy2m2 = np.mean(np.clip((y2)**2, 0, Delta**2))
    py2m2 = cy2m2 + np.random.normal(0, math.sqrt(Delta**4/(2*rho*n2**2)))
    py2m = float(n1)/n * py2m1 + float(n2)/n * py2m2

    # slopes
    pb1 = pxym1/px2m1
    pb2 = pxym2/px2m2
    pb = float(n1)/n * pb1 + float(n2)/n * pb2

    # standard errors
    ps2_0 = (n*py2m + n*(pb)**2*px2m - 2*n*pxym2*pb)/(n-2)
    ps2 = (n1*py2m1 + n1*(pb1)**2*px2m1 - 2*n1*pxym1*pb1 + n2*py2m2 + n2*(pb2)**2*px2m2 - 2*n2*pxym2*pb2)/(n-2)

    # test statistic
    pf_stat = (n1*px2m1*n2*px2m2)*(pb1-pb2)**2/(ps2*n*px2m)

    return (pxm1, pxm2, pxm, px2m1, px2m2, px2m, pb1, pb2, ps2_0, ps2, pf_stat)

# non-DP test statistic for testing mixtures via Kruskal-Wallis
def np_mix_kw(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    n = n1 + n2
    sranks = ranks(np.concatenate([s1, s2]))
    r1 = np.mean(sranks[0:len(s1)])
    r2 = np.mean(sranks[len(s1):])
    h = 4*(n-1)/n**2 * (n1*abs(r1 - (n+1)/2) + n2*abs(r2 - (n+1)/2))
    return h

# test statistic for testing mixtures via Kruskal-Wallis
def mix_kw(s1, s2, rho):
    h = np_mix_kw(s1, s2)

    return h + np.random.normal(0, math.sqrt(8**2/(2*rho)))

################################ Monte Carlo Testing ################################

# Monte Carlo DP test for testing linear relationship
def mc_lin_norm(ym, b1, b2, x2m, s2_0, xm, t, tp, n, rho, Delta, alpha, sigma_e, stat):
    K = 10/alpha
    ts = []

    for k in range(1, int(K)+1):
        xk = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
        yk = np.random.normal(ym, math.sqrt(s2_0), n)
        (xmk, x2mk, b1k, b2k, s2_0k, s2k, f_statk) = stat(xk, yk, n)
        ts.append(f_statk)

    ts.sort()
    ts = np.array(ts)
    r = math.ceil((K+1)*(1-alpha))

    return (t > ts[r], tp > ts[r])

# Monte Carlo DP OLS Test for testing mixtures via F-statistic
def mc_mix(b1, b2, x2m1, x2m2, x2m, s2_0, xm1, xm2, xm, t, tp,
           n1, n, rho, Delta, alpha, sigma_e, stat):
    n1 = int(n1)
    n2 = n-n1
    K = 10/alpha
    ts = []

    for k in range(1, int(K)+1):
        xk = np.random.normal(xm,
                              math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
        xk1 = xk[0:n1]
        xk2 = xk[n1:]

        yk1 = np.random.normal(b1*xk1, math.sqrt(s2_0))
        yk2 = np.random.normal(b1*xk2, math.sqrt(s2_0))
        yk = np.concatenate([yk1, yk2])
        (xm1k, xm2k, xmk, x2m1k, x2m2k, x2mk, b1k, b2k, s2_0k, s2k, f_statk) = stat(xk, yk, n1, n)
        ts.append(f_statk)

    ts.sort()
    ts = np.array(ts)
    r = math.ceil((K+1)*(1-alpha))

    return (t > ts[r], tp > ts[r])

# Monte Carlo DP OLS Test for testing mixtures via Kruskal-Wallis
def mc_mix_kw(t, tp, n1, n, rho, Delta, alpha, sigma_e, Clip, stat_kw):
    n1 = int(n1)
    n2 = n-n1
    K = 10/alpha
    ts = []

    for k in range(1, int(K)+1):
        s1 = np.random.uniform(-Clip, Clip, n1)
        s2 = np.random.uniform(-Clip, Clip, n2)
        tk = stat_kw(s1, s2)
        ts.append(tk)

    ts.sort()
    ts = np.array(ts)
    r = math.ceil((K+1)*(1-alpha))

    return (t > ts[r], tp > ts[r])

# m is a real number
# b is a number between 0 and 1
# q is a number between 0 and 1
def Tulap(m, b, q):
    g1, g2 = np.random.geometric(1-b, 2)
    U = np.random.uniform(-1.0/2, 1.0/2, 1)
    N = g1 - g2 + U + m
    # to fill up
    return N

# Monte Carlo DP OLS Test for testing linear relationship via the Tulap distribution
def mc_lin_norm_via_Tulap(x, y, n, rho, alpha):
    K = 10/alpha
    ts = []

    for k in range(1, int(K)+1):
        xk = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
        yk = np.random.normal(ym, math.sqrt(s2_0), n)
        (xmk, x2mk, b1k, b2k, s2_0k, s2k, f_statk) = stat(xk, yk, n)
        ts.append(f_statk)

    ts.sort()
    ts = np.array(ts)
    r = math.ceil((K+1)*(1-alpha))

    return (t > ts[r], tp > ts[r])

# Monte Carlo DP OLS Test for testing linear relationship via CI
def mc_lin_norm_via_CI(x, y, n, rho, alpha):
    K = 10/alpha

    X = np.transpose(np.vstack((np.ones(n), x)))

    args = {}
    args['num_bootstraps'] = int(K)
    args['rho'] = rho
    args['gamma'] = 4
    args['delta'] = 4
    args['zeta'] = 4
    args['D'] = 2
    args['N'] = n

    (beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv) = private_OLS(X, y, args)
    slopes = hybrid_bootstrap_OLS(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, args)
    slopes.sort()
    r = math.ceil((K+1)*(1-alpha/2))
    l = math.ceil((K+1)*(alpha/2))

    in_interval = slopes[l] <= 0 <= slopes[r]

    return not(in_interval)

##
## Modified from https://github.com/ceciliaferrando/PB-DP-CIs/blob/master/DPCIs-OLS.py
##
def private_OLS(X, y, args):
    """
    :param X: independent variable data
    :param y: dependent variable data
    :param U: errors
    :param beta_true: true beta coefficient
    :param args: arguments
    :return: private estimates of beta, Q, sigma; sensitivity of XtX, sensitivity of XtY
    """

    D = args['D']
    gamma = args['gamma']
    zeta = args['zeta']
    rho = args['rho']
    N = args['N']
    w, V = compute_dp_noise(D, gamma, zeta, rho)
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)

    beta_hat = np.dot(inv(XtX), Xty)  # this is the non-private estimate for beta
    beta_hat_priv = np.dot(inv(XtX + V), (Xty + w))

    Q_hat = 1 / N * XtX
    Q_hat_priv = Q_hat + 1 / N * V
    if is_pos_def(Q_hat_priv) == False:
        Q_hat_priv = make_pos_def(Q_hat_priv, small_positive=0.1)

    upper_bound_x, lower_bound_x = gamma, -gamma
    upper_bound_y, lower_bound_y = zeta, -zeta
    width_term = max((upper_bound_y - np.sum(lower_bound_x * np.abs(beta_hat))) ** 2,
                     (lower_bound_y - np.sum(upper_bound_x * np.abs(beta_hat))) ** 2)
    Delta_sigma_sq = 1.0 / (N - D) * width_term
    sigma_sq_hat_priv = 1.0 / (N - D) * np.sum((y - np.dot(X, beta_hat)) ** 2) + \
                        np.random.normal(0, math.sqrt(Delta_sigma_sq**2/(2*rho/3)), 1) ### note ##

    if sigma_sq_hat_priv < 0:
        sigma_sq_hat_priv = 0.1

    return (beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv)

def hybrid_bootstrap_OLS(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, args):
    """
    :param beta_hat_priv: private estimate of beta
    :param Q_hat_priv: private estimate of Q
    :param sigma_sq_hat_priv: private estimate of sigma**2
    :param Delta_XX: global sensitivity of XtX
    :param Delta_XY: global sesitivity of XtY
    :param args: input arguments
    :return: vector of bootstrap beta estimates
    """

    D = args['D']
    gamma = args['gamma']
    zeta = args['zeta']
    rho = args['rho']
    N = args['N']
    num_bootstraps = args['num_bootstraps']

    beta_star_vec = []

    for b in range(num_bootstraps):

        w_star = compute_dp_noise(D, gamma, zeta, rho)[0]
        V_star = compute_dp_noise(D, gamma, zeta, rho)[1]

        cov_matrix = sigma_sq_hat_priv * Q_hat_priv
        # if is_pos_def(cov_matrix) == False:
        #     cov_matrix = make_pos_def(cov_matrix)
        Z_star = np.random.multivariate_normal(np.zeros(D), cov_matrix)

        Q_hat_star = Q_hat_priv + 1 / N * V_star

        beta_star_b = np.dot(np.dot(inv(Q_hat_star), Q_hat_priv), beta_hat_priv) + \
                      np.dot(inv(Q_hat_star), 1 / np.sqrt(N) * Z_star + 1 / N * w_star)
        beta_star_vec.append(beta_star_b[1])

    return beta_star_vec

def compute_dp_noise(D, gamma, zeta, rho):
    '''
    :param gs_XY: global sensitivity computed on XtY
    :param gs_XX: global sensitivity computed on XtX
    :param D: dimension of ols
    :return: w, the normal noise to be added on XY term, a D-dim vector
             v, the normal noise to be added on XX term, a D-D matrix
    '''
    Delta_w = D * 2*gamma * zeta
    w = np.zeros((D,))
    for i_w in range(len(w)):
        w[i_w] = np.random.normal(0, math.sqrt(Delta_w**2/(2*rho/3)), 1)

    Delta_V = D*(D + 1) * gamma**2
    V = np.zeros((D, D))
    for i_x in range(D):
        for j_x in range(D):
            if i_x <= j_x:
                V[i_x][j_x] = np.random.normal(0, math.sqrt(Delta_V**2/(2*rho/3)), 1)
                V[j_x][i_x] = V[i_x][j_x]
    return w, V

def is_pos_def(x):
    """
    :param x: input matrix
    :return: True if x is PSD
    """
    return np.all(np.linalg.eigvals(x) > 0)

def make_pos_def(x, small_positive=0.1):
    """
    :param x: input matrix
    :param small_positive: float
    :return: PSD projection of x
    """

    # Compute SVD and eigenvalue decompositions
    (u, s, v) = np.linalg.svd(x)
    (l, w) = np.linalg.eig(x)

    # Make sure x is not PSD
    if np.all(l >= 0):
        raise ValueError("X is already PSD")
    l_prime = np.where(l > 0, l, small_positive)
    xPSD = w.dot(np.diag(l_prime)).dot(w.T)

    # Check
    ll, _ = np.linalg.eig(xPSD)
    assert (np.all(ll > 0))

    return xPSD
