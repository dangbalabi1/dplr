#!/usr/bin/env python3

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import dp_lr_testing as test
import importlib
importlib.reload(test)

random.seed(2021)

colors = ['tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:brown',
          'tab:pink', 'tab:gray', 'tab:olive',
          'tab:cyan']


################################ Data Generation Utilities ################################

def gen_data_lin_norm(n, varx, barx, vare, slope, intercept):
    n = int(n)
    x = np.random.normal(barx, math.sqrt(varx), n)
    x = np.array(x)
    xm = np.mean(x)
    y = np.random.normal(slope*(x-xm) + intercept, math.sqrt(vare))
    y = np.array(y)
    ym = np.mean(y)
    return x, y, xm, ym, n

def gen_data_lin_unif(n, low, high, vare, slope, intercept):
    n = int(n)
    x = test.unifx(low, high, n)
    x = np.array(x)
    xm = np.mean(x)
    y = np.random.normal(slope*(x-xm) + intercept, math.sqrt(vare))
    y = np.array(y)
    ym = np.mean(y)
    return x, y, xm, ym, n

def gen_data_lin_exp(n, scale, vare, slope, intercept):
    n = int(n)
    x = test.expx(scale, n)
    x = np.array(x)
    xm = np.mean(x)
    y = np.random.normal(slope*(x-xm) + intercept, math.sqrt(vare))
    y = np.array(y)
    ym = np.mean(y)
    return x, y, xm, ym, n

def gen_data_mix(n, frac, varx, barx, vare, slope1, slope2):
    n1 = int(frac*n)
    x = np.random.normal(barx, math.sqrt(varx), n)
    x = np.array(x)
    xm = np.mean(x)
    y1 = np.random.normal(slope1*x[:n1], math.sqrt(vare))
    y2 = np.random.normal(slope2*x[n1:], math.sqrt(vare))
    y = np.concatenate([y1, y2])
    y = np.array(y)
    ym = np.mean(y)
    return x, y, xm, ym, n

def gen_data_mix_kw(n, frac, varx, barx, vare, slope1, slope2):
    x, y, xm, ym, n = gen_data_mix(n, frac, varx, barx, vare, slope1, slope2)
    n1 = int(frac*n)
    n2 = n-n1
    x1 = x[0:n1]
    x2 = x[n1:]
    y1 = y[0:n1]
    y2 = y[n1:]

    s1 = []
    x1_y1 = list(zip(x1, y1))
    if n1%2 == 1: n1 = n1-1
    perm1 = np.random.choice(n1, size=n1, replace=False)
    for l in range(0, n1, 2):
        i = perm1[l]
        j = perm1[l+1]
        x1_1, y1_1 = x1_y1[i]
        x1_2, y1_2 = x1_y1[j]
        if x1_1 != x1_2: s1.append(float(y1_2-y1_1)/(x1_2-x1_1))
    s2 = []
    x2_y2 = list(zip(x2, y2))
    if n2%2 == 1: n2 = n2-1
    perm2 = np.random.choice(n2, size=n2, replace=False)
    for l in range(0, n2, 2):
        i = perm2[l]
        j = perm2[l+1]
        x2_1, y2_1 = x2_y2[i]
        x2_2, y2_2 = x2_y2[j]
        if x2_1 != x2_2: s2.append(float(y2_2-y2_1)/(x2_2-x2_1))
    return s1, s2

def get_sizes():
    sizes = np.arange(0, 8001, 1000)
    sizes[0] = 300
    return sizes

################################ Linear Relationship ################################

def linear(slope, sigma_e, Delta, gen_data, tester, other_string, varx, barx):
    print("Linear Tester: slope={0}, sigma_e={1}, Delta={2}, varx={3}, barx={4}".format(slope, sigma_e, Delta, varx, barx))
    r = p = 2
    q = 1
    alpha = 0.05
    num_trials = 1000

    powers = [[], [], [], [], []]
    pvals = [[], [], [], [], []]
    rhos = [(0.1)**2/2, (1)**2/2, (5)**2/2, (10)**2/2]

    for n in get_sizes():
        # private
        for i in range(len(rhos)):
            print("Running for rho={0}, n={1}".format(rhos[i], n))
            rho = rhos[i]
            power = 0
            pval = 0
            for j in range(num_trials):
                x, y, xm, ym, n = gen_data(n, sigma_e, slope)
                stat = lambda x, y, n : test.lin(x, y, n, rho, Delta)
                (xm, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n)
                (rej, rejn) = (False, False)
                if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                    xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                    yp = np.random.normal(b2, math.sqrt(s2_0), n)
                    (xmp, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n)
                    (rej, rejn) = tester(ym, b1, b2, x2m, s2_0, xm, t, tp, n, rho, Delta, alpha, sigma_e, stat)
                power += 1 if rej else 0
                pval += 1 if rejn else 0
            power /= float(num_trials)
            pval /= float(num_trials)
            powers[i].append(power)
            pvals[i].append(pval)

        # non-private
        power = 0
        pval = 0
        for j in range(num_trials):
            x, y, xm, ym, n = gen_data(n, sigma_e, slope)
            stat = lambda x, y, n : test.np_lin(x, y, n)
            (xm, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n)
            (rej, rejn) = (False, False)
            if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                yp = np.random.normal(b2, math.sqrt(s2_0), n)
                (xmp, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n)
                (rej, rejn) = tester(ym, b1, b2, x2m, s2_0, xm, t, tp, n, rho, Delta, alpha, sigma_e, stat)
            power += 1 if rej else 0
            pval += 1 if rejn else 0
        power /= float(num_trials)
        pval /= float(num_trials)
        powers[len(powers)-1].append(power)
        pvals[len(pvals)-1].append(pval)

    print(powers)
    print(pvals)
    print(rhos)
    np.save("thesis_vectors{0}/linear_{1}_{2}_{3}_{4}_power.npy".format(num_trials, other_string, slope, sigma_e, Delta), powers)
    np.save("thesis_vectors{0}/linear_{1}_{2}_{3}_{4}_pval.npy".format(num_trials, other_string, slope, sigma_e, Delta), pvals)
    np.save("thesis_vectors{0}/linear_{1}_{2}_{3}_{4}_rhos.npy".format(num_trials, other_string, slope, sigma_e, Delta), rhos)

    ns = get_sizes()
    powers1 = np.array(powers[0])
    powers2 = np.array(powers[1])
    powers3 = np.array(powers[2])
    powers4 = np.array(powers[3])
    powers5 = np.array(powers[4])
    plt.plot(ns, powers1, label=r"$\rho = (0.1)^2/2$")
    plt.plot(ns, powers2, label=r"$\rho = (1)^2/2$")
    plt.plot(ns, powers3, label=r"$\rho = (5)^2/2$")
    plt.plot(ns, powers4, label=r"$\rho = (10)^2/2$")
    plt.plot(ns, powers5, label=r"non-private")
    plt.xlabel(r"$n$")
    plt.ylabel(r"Power")
    plt.legend()
    plt.savefig("thesis_images{0}/linear_{1}_{2}_{3}_{4}_power.png".format(num_trials, other_string, slope, sigma_e, Delta))
    plt.gcf().clear()

    ns = get_sizes()
    pvals1 = np.array(pvals[0])
    pvals2 = np.array(pvals[1])
    pvals3 = np.array(pvals[2])
    pvals4 = np.array(pvals[3])
    pvals5 = np.array(pvals[4])
    plt.plot(ns, pvals1, label=r"$\rho = (0.1)^2/2$")
    plt.plot(ns, pvals2, label=r"$\rho = (1)^2/2$")
    plt.plot(ns, pvals3, label=r"$\rho = (5)^2/2$")
    plt.plot(ns, pvals4, label=r"$\rho = (10)^2/2$")
    plt.plot(ns, pvals5, label=r"non-private")
    plt.xlabel(r"$n$")
    plt.ylabel(r"p-value")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("thesis_images{0}/linear_{1}_{2}_{3}_{4}_pval.png".format(num_trials, other_string, slope, sigma_e, Delta))
    plt.gcf().clear()

def linear_CI(slope, sigma_e, Delta, gen_data, tester1, tester2, other_string, varx, barx):
    print("Linear Tester with CI: slope={0}, sigma_e={1}, Delta={2}, varx={3}, barx={4}".format(slope, sigma_e, Delta, varx, barx))
    r = p = 2
    q = 1
    alpha = 0.05
    num_trials = 2000
    powers1 = [[], [], []]
    powers2 = [[], [], []]
    pvals1 = [[], [], []]
    pvals2 = [[], [], []]
    rhos = [(1)**2/2, (2)**2/2, (3)**2/2]
    ns = [500, 600, 700, 800, 900, 1000]

    for n in ns:
        # private
        for i in range(len(rhos)):
            print("Running for rho={0}, n={1}".format(rhos[i], n))
            rho = rhos[i]
            power1 = 0
            power2 = 0
            pval1 = 0
            pval2 = 0
            for j in range(num_trials):
                x, y, xm, ym, n = gen_data(n, sigma_e, slope)
                stat = lambda x, y, n : test.lin(x, y, n, rho, Delta)
                (xm, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n)
                (rej1, rejn1) = (False, False)
                if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                    xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                    yp = np.random.normal(b2, math.sqrt(s2_0), n)
                    (xmp, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n)
                    (rej1, rejn1) = tester1(ym, b1, b2, x2m, s2_0, xm, t, tp, n, rho, Delta, alpha, sigma_e, stat)
                (rej2, rejn2) = (False, False)
                rej2 = tester2(x, y, n, rho, alpha)
                x, y, xm, ym, n = gen_data(n, sigma_e, 0)
                rejn2 = tester2(x, y, n, rho, alpha)
                power1 += 1 if rej1 else 0
                power2 += 1 if rej2 else 0
                pval1 += 1 if rejn1 else 0
                pval2 += 1 if rejn2 else 0
            power1 /= float(num_trials)
            power2 /= float(num_trials)
            pval1 /= float(num_trials)
            pval2 /= float(num_trials)
            powers1[i].append(power1)
            powers2[i].append(power2)
            pvals1[i].append(pval1)
            pvals2[i].append(pval2)

    print("Powers 1: ", powers1)
    print("Powers 2: ", powers2)
    print("Significance 1: ", pvals1)
    print("Significance 2: ", pvals2)
    print(rhos)

    powers1_f = np.array(powers1[0])
    powers2_f = np.array(powers1[1])
    powers3_f = np.array(powers1[2])
    powers1_ci = np.array(powers2[0])
    powers2_ci = np.array(powers2[1])
    powers3_ci = np.array(powers2[2])
    plt.plot(ns, powers1_f, label=r"F-stat $\rho = (1)^2/2$", color=colors[0])
    plt.plot(ns, powers2_f, label=r"F-stat $\rho = (2)^2/2$", color=colors[1])
    plt.plot(ns, powers3_f, label=r"F-stat $\rho = (3)^2/2$", color=colors[2])
    plt.plot(ns, powers1_ci, label=r"CI $\rho = (1)^2/2$", color=colors[0], linestyle='dashed')
    plt.plot(ns, powers2_ci, label=r"CI $\rho = (2)^2/2$", color=colors[1], linestyle='dashed')
    plt.plot(ns, powers3_ci, label=r"CI $\rho = (3)^2/2$", color=colors[2], linestyle='dashed')

    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"Power", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("thesis_images{0}/linear_ci_{1}_{2}_{3}_{4}_{5}_{6}_power.png".format(num_trials, other_string, slope, sigma_e, Delta, varx, barx))
    plt.gcf().clear()

    pvals1_f = np.array(pvals1[0])
    pvals2_f = np.array(pvals1[1])
    pvals3_f = np.array(pvals1[2])
    pvals1_ci = np.array(pvals2[0])
    pvals2_ci = np.array(pvals2[1])
    pvals3_ci = np.array(pvals2[2])
    plt.plot(ns, pvals1_f, label=r"F-stat $\rho = (1)^2/2$", color=colors[0])
    plt.plot(ns, pvals2_f, label=r"F-stat $\rho = (2)^2/2$", color=colors[1])
    plt.plot(ns, pvals3_f, label=r"F-stat $\rho = (3)^2/2$", color=colors[2])
    plt.plot(ns, pvals1_ci, label=r"CI $\rho = (1)^2/2$", color=colors[0], linestyle='dashed')
    plt.plot(ns, pvals2_ci, label=r"CI $\rho = (2)^2/2$", color=colors[1], linestyle='dashed')
    plt.plot(ns, pvals3_ci, label=r"CI $\rho = (3)^2/2$", color=colors[2], linestyle='dashed')

    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"Significance", fontsize=20)
    plt.ylim(0, 0.1)
    plt.yticks(np.arange(0, 0.21, 0.025))
    plt.legend(fontsize=15)
    plt.savefig("thesis_images{0}/linear_ci_{1}_{2}_{3}_{4}_{5}_{6}_pval.png".format(num_trials, other_string, slope, sigma_e, Delta, varx, barx))
    plt.gcf().clear()

def linear_Tulap(slope, sigma_e, Delta, gen_data, tester1, tester2, other_string, varx, barx):
    print("Linear Tester with Tulap: slope={0}, sigma_e={1}, Delta={2}, varx={3}, barx={4}".format(slope, sigma_e, Delta, varx, barx))
    r = p = 2
    q = 1
    alpha = 0.05
    num_trials = 2000
    powers1 = [[], [], []]
    powers2 = [[], [], []]
    pvals1 = [[], [], []]
    pvals2 = [[], [], []]
    rhos = [(0.1)**2/2, (1)**2/2, (2)**2/2]
    ns = [500, 600, 700, 800, 900, 1000]

    for n in ns:
        # private
        for i in range(len(rhos)):
            print("Running for rho={0}, n={1}".format(rhos[i], n))
            rho = rhos[i]
            power1 = 0
            power2 = 0
            pval1 = 0
            pval2 = 0
            for j in range(num_trials):
                x, y, xm, ym, n = gen_data(n, sigma_e, slope)
                stat = lambda x, y, n : test.lin(x, y, n, rho, Delta)
                (xm, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n)
                (rej1, rejn1) = (False, False)
                if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                    xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                    yp = np.random.normal(b2, math.sqrt(s2_0), n)
                    (xmp, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n)
                    (rej1, rejn1) = tester1(ym, b1, b2, x2m, s2_0, xm, t, tp, n, rho, Delta, alpha, sigma_e, stat)
                (rej2, rejn2) = (False, False)
                rej2 = tester2(x, y, n, rho, alpha)
                x, y, xm, ym, n = gen_data(n, sigma_e, 0)
                rejn2 = tester2(x, y, n, rho, alpha)
                power1 += 1 if rej1 else 0
                power2 += 1 if rej2 else 0
                pval1 += 1 if rejn1 else 0
                pval2 += 1 if rejn2 else 0
            power1 /= float(num_trials)
            power2 /= float(num_trials)
            pval1 /= float(num_trials)
            pval2 /= float(num_trials)
            powers1[i].append(power1)
            powers2[i].append(power2)
            pvals1[i].append(pval1)
            pvals2[i].append(pval2)

    print("Powers 1: ", powers1)
    print("Powers 2: ", powers2)
    print("Significance 1: ", pvals1)
    print("Significance 2: ", pvals2)
    print(rhos)

    powers1_f = np.array(powers1[0])
    powers2_f = np.array(powers1[1])
    powers3_f = np.array(powers1[2])
    powers1_ci = np.array(powers2[0])
    powers2_ci = np.array(powers2[1])
    powers3_ci = np.array(powers2[2])
    plt.plot(ns, powers1_f, label=r"F-stat $\rho = (1)^2/2$", color=colors[0])
    plt.plot(ns, powers2_f, label=r"F-stat $\rho = (2)^2/2$", color=colors[1])
    plt.plot(ns, powers3_f, label=r"F-stat $\rho = (3)^2/2$", color=colors[2])
    plt.plot(ns, powers1_ci, label=r"CI $\rho = (1)^2/2$", color=colors[0], linestyle='dashed')
    plt.plot(ns, powers2_ci, label=r"CI $\rho = (2)^2/2$", color=colors[1], linestyle='dashed')
    plt.plot(ns, powers3_ci, label=r"CI $\rho = (3)^2/2$", color=colors[2], linestyle='dashed')

    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"Power", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("thesis_images{0}/linear_ci_{1}_{2}_{3}_{4}_{5}_{6}_power.png".format(num_trials, other_string, slope, sigma_e, Delta, varx, barx))
    plt.gcf().clear()

    pvals1_f = np.array(pvals1[0])
    pvals2_f = np.array(pvals1[1])
    pvals3_f = np.array(pvals1[2])
    pvals1_ci = np.array(pvals2[0])
    pvals2_ci = np.array(pvals2[1])
    pvals3_ci = np.array(pvals2[2])
    plt.plot(ns, pvals1_f, label=r"F-stat $\rho = (1)^2/2$", color=colors[0])
    plt.plot(ns, pvals2_f, label=r"F-stat $\rho = (2)^2/2$", color=colors[1])
    plt.plot(ns, pvals3_f, label=r"F-stat $\rho = (3)^2/2$", color=colors[2])
    plt.plot(ns, pvals1_ci, label=r"CI $\rho = (1)^2/2$", color=colors[0], linestyle='dashed')
    plt.plot(ns, pvals2_ci, label=r"CI $\rho = (2)^2/2$", color=colors[1], linestyle='dashed')
    plt.plot(ns, pvals3_ci, label=r"CI $\rho = (3)^2/2$", color=colors[2], linestyle='dashed')

    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"Significance", fontsize=20)
    plt.ylim(0, 0.1)
    plt.yticks(np.arange(0, 0.21, 0.025))
    plt.legend(fontsize=15)
    plt.savefig("thesis_images{0}/linear_ci_{1}_{2}_{3}_{4}_{5}_{6}_pval.png".format(num_trials, other_string, slope, sigma_e, Delta, varx, barx))
    plt.gcf().clear()

################################ Mixture Models ################################
def mixture(slope1, slope2, frac, sigma_e, Delta, varx, barx):
    print("Mixture Tester: slope1={0}, slope2={1}, frac={2}, sigma_e={3}, Delta={4}, varx={5}, barx={6}".format(slope1, slope2, frac, sigma_e, Delta, varx, barx))
    r = p = 2
    q = 1
    alpha = 0.05
    num_trials = 1000

    powers = [[], [], [], [], []]
    pvals = [[], [], [], [], []]
    rhos = [(0.1)**2/2, (1)**2/2, (5)**2/2, (10)**2/2]

    for n in get_sizes():
        x, y, xm, ym, n = gen_data_mix(n, frac, varx, barx, sigma_e**2, slope1, slope2)
        # private
        for i in range(len(rhos)):
            print("Running for rho={0}, n={1}".format(rhos[i], n))
            rho = rhos[i]
            power = 0
            pval = 0
            for j in range(num_trials):
                x, y, xm, ym, n = gen_data_mix(n, frac, varx, barx, sigma_e**2, slope1, slope2)
                n1 = int(frac*n)
                n2 = n-n1
                stat = lambda x, y, n1, n: test.mix(x, y, n1, n, rho, Delta)
                (xm1, xm2, xm, x2m1, x2m2, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n1, n)
                (rej, rejn) = (False, False)
                if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                    xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                    xp1 = xp[0:n1]
                    xp2 = xp[n1:]
                    yp1 = np.random.normal(b1*xp1, math.sqrt(s2_0))
                    yp2 = np.random.normal(b1*xp2, math.sqrt(s2_0))
                    yp = np.concatenate([yp1, yp2])
                    (xm1p, xm2p, xmp, x2m1p, x2m2p, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n1, n)
                    (rej, rejn) = test.mc_mix(b1, b2, x2m1, x2m2, x2m, s2_0, xm1, xm2, xm, t, tp, n1, n, rho, Delta, alpha, sigma_e, stat)
                power += 1 if rej else 0
                pval += 1 if rejn else 0

            power /= float(num_trials)
            pval /= float(num_trials)
            powers[i].append(power)
            pvals[i].append(pval)

        # non-private
        power = 0
        pval = 0
        for j in range(num_trials):
            x, y, xm, ym, n = gen_data_mix(n, frac, varx, barx, sigma_e**2, slope1, slope2)
            n1 = int(frac*n)
            n2 = n-n1
            stat = lambda x, y, n1, n: test.np_mix(x, y, n1, n)
            (xm1, xm2, xm, x2m1, x2m2, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n1, n)
            (rej, rejn) = (False, False)
            if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                xp1 = xp[0:n1]
                xp2 = xp[n1:]
                yp1 = np.random.normal(b1*xp1, math.sqrt(s2_0))
                yp2 = np.random.normal(b1*xp2, math.sqrt(s2_0))
                yp = np.concatenate([yp1, yp2])
                (xm1p, xm2p, xmp, x2m1p, x2m2p, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n1, n)
                (rej, rejn) = test.mc_mix(b1, b2, x2m1, x2m2, x2m, s2_0, xm1, xm2, xm, t, tp, n1, n, rho, Delta, alpha, sigma_e, stat)
            power += 1 if rej else 0
            pval += 1 if rejn else 0
    
        power /= float(num_trials)
        pval /= float(num_trials)
        powers[len(powers)-1].append(power)
        pvals[len(pvals)-1].append(pval)

    print(powers)
    print(pvals)
    print(rhos)
    np.save("thesis_vectors{0}/mix_{1}_{2}_{3}_{4}_{5}_power.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta), powers)
    np.save("thesis_vectors{0}/mix_{1}_{2}_{3}_{4}_{5}_pval.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta), pvals)
    np.save("thesis_vectors{0}/mix_{1}_{2}_{3}_{4}_{5}_rhos.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta), rhos)

    ns = get_sizes()
    powers1 = np.array(powers[0])
    powers2 = np.array(powers[1])
    powers3 = np.array(powers[2])
    powers4 = np.array(powers[3])
    powers5 = np.array(powers[4])
    plt.plot(ns, powers1, label=r"$\rho = (0.1)^2/2$")
    plt.plot(ns, powers2, label=r"$\rho = (1)^2/2$")
    plt.plot(ns, powers3, label=r"$\rho = (5)^2/2$")
    plt.plot(ns, powers4, label=r"$\rho = (10)^2/2$")
    plt.plot(ns, powers5, label=r"non-private")
    plt.xlabel(r"$n$")
    plt.ylabel(r"Power")
    plt.legend()
    plt.savefig("thesis_images{0}/mix_{1}_{2}_{3}_{4}_{5}_power.png".format(num_trials, slope1, slope2, frac, sigma_e, Delta))
    plt.gcf().clear()

    ns = get_sizes()
    pvals1 = np.array(pvals[0])
    pvals2 = np.array(pvals[1])
    pvals3 = np.array(pvals[2])
    pvals4 = np.array(pvals[3])
    pvals5 = np.array(pvals[4])
    plt.plot(ns, pvals1, label=r"$\rho = (0.1)^2/2$")
    plt.plot(ns, pvals2, label=r"$\rho = (1)^2/2$")
    plt.plot(ns, pvals3, label=r"$\rho = (5)^2/2$")
    plt.plot(ns, pvals4, label=r"$\rho = (10)^2/2$")
    plt.plot(ns, pvals5, label=r"non-private")
    plt.xlabel(r"$n$")
    plt.ylabel(r"p-value")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("thesis_images{0}/mix_{1}_{2}_{3}_{4}_{5}_pval.png".format(num_trials, slope1, slope2, frac, sigma_e, Delta))
    plt.gcf().clear()

# todo: move from 100s to 500 max
def mixture_kw(slope1, slope2, frac, sigma_e, Delta, Clip, varx, barx):
    print("Mixture Tester for KW: slope1={0}, slope2={1}, frac={2}, sigma_e={3}, Delta={4}, Clip={5}, varx={6}".format(slope1, slope2, frac, sigma_e, Delta, Clip, varx))
    r = p = 2
    q = 1
    alpha = 0.05
    num_trials = 1000

    powers = [[], [], [], []]
    pvals = [[], [], [], []]
    powers_kw = [[], [], [], []]
    pvals_kw = [[], [], [], []]
    rhos = [(0.5)**2/2, (1)**2/2, (5)**2/2]
    ns = [100, 200, 300, 400, 500]

    for n in ns:
        # private
        for i in range(len(rhos)):
            print("Running for rho={0}, n={1}".format(rhos[i], n))
            rho = rhos[i]
            power = 0
            pval = 0
            power_kw = 0
            pval_kw = 0
            for j in range(num_trials):
                x, y, xm, ym, n = gen_data_mix(n, frac, varx, barx, sigma_e**2, slope1, slope2)
                n1 = int(frac*n)
                n2 = n-n1
                stat = lambda x, y, n1, n: test.mix(x, y, n1, n, rho, Delta)
                (xm1, xm2, xm, x2m1, x2m2, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n1, n)
                (rej, rejn) = (False, False)
                if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                    xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                    xp1 = xp[0:n1]
                    xp2 = xp[n1:]
                    yp1 = np.random.normal(b1*xp1, math.sqrt(s2_0))
                    yp2 = np.random.normal(b1*xp2, math.sqrt(s2_0))
                    yp = np.concatenate([yp1, yp2])
                    (xm1p, xm2p, xmp, x2m1p, x2m2p, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n1, n)
                    (rej, rejn) = test.mc_mix(b1, b2, x2m1, x2m2, x2m, s2_0, xm1, xm2, xm, t, tp, n1, n, rho, Delta, alpha, sigma_e, stat)
                power += 1 if rej else 0
                pval += 1 if rejn else 0

                s1, s2 = gen_data_mix_kw(n, frac, varx, barx, sigma_e**2, slope1, slope2)
                n1 = len(s1)
                n2 = len(s2)
                s1_n = np.random.uniform(-Clip, Clip, n1)
                s2_n = np.random.uniform(-Clip, Clip, n2)
                t = test.mix_kw(s1, s2, rho)
                tp = test.mix_kw(s1_n, s2_n, rho)
                stat_kw = lambda s1, s2 : test.mix_kw(s1, s2, rho)
                (rej_kw, rejn_kw) = test.mc_mix_kw(t, tp, n1, n, rho, Delta, alpha, sigma_e, Clip, stat_kw)
                power_kw += 1 if rej_kw else 0
                pval_kw += 1 if rejn_kw else 0

            power /= float(num_trials)
            pval /= float(num_trials)
            power_kw /= float(num_trials)
            pval_kw /= float(num_trials)
            powers[i].append(power)
            pvals[i].append(pval)
            powers_kw[i].append(power_kw)
            pvals_kw[i].append(pval_kw)

        # non-private
        power = 0
        pval = 0
        power_kw = 0
        pval_kw = 0
        for j in range(num_trials):
            x, y, xm, ym, n = gen_data_mix(n, frac, varx, barx, sigma_e**2, slope1, slope2)
            n1 = int(frac*n)
            n2 = n-n1
            stat = lambda x, y, n1, n: test.np_mix(x, y, n1, n)
            (xm1, xm2, xm, x2m1, x2m2, x2m, b1, b2, s2_0, s2, t) = stat(x, y, n1, n)
            (rej, rejn) = (False, False)
            if min(s2_0, (n*x2m - n*xm**2)/(n-1)) > 0:
                xp = np.random.normal(xm, math.sqrt((n*x2m - n*xm**2)/(n-1)), n)
                xp1 = xp[0:n1]
                xp2 = xp[n1:]
                yp1 = np.random.normal(b1*xp1, math.sqrt(s2_0))
                yp2 = np.random.normal(b1*xp2, math.sqrt(s2_0))
                yp = np.concatenate([yp1, yp2])
                (xm1p, xm2p, xmp, x2m1p, x2m2p, x2mp, b1p, b2p, s2_0p, s2p, tp) = stat(xp, yp, n1, n)
                (rej, rejn) = test.mc_mix(b1, b2, x2m1, x2m2, x2m, s2_0, xm1, xm2, xm, t, tp, n1, n, rho, Delta, alpha, sigma_e, stat)
            power += 1 if rej else 0
            pval += 1 if rejn else 0

            s1, s2 = gen_data_mix_kw(n, frac, varx, barx, sigma_e**2, slope1, slope2)
            n1 = len(s1)
            n2 = len(s2)
            s1_n = np.random.uniform(-Clip, Clip, n1)
            s2_n = np.random.uniform(-Clip, Clip, n2)
            t = test.np_mix_kw(s1, s2)
            tp = test.np_mix_kw(s1_n, s2_n)
            stat_kw = lambda s1, s2 : test.np_mix_kw(s1, s2)
            (rej_kw, rejn_kw) = test.mc_mix_kw(t, tp, n1, n, rho, Delta, alpha, sigma_e, Clip, stat_kw)
            power_kw += 1 if rej_kw else 0
            pval_kw += 1 if rejn_kw else 0

        power /= float(num_trials)
        pval /= float(num_trials)
        power_kw /= float(num_trials)
        pval_kw /= float(num_trials)
        powers[len(powers)-1].append(power)
        pvals[len(pvals)-1].append(pval)
        powers_kw[len(powers_kw)-1].append(power_kw)
        pvals_kw[len(pvals_kw)-1].append(pval_kw)


    print(powers)
    print(pvals)
    print("KW============================================")
    print(powers_kw)
    print(pvals_kw)
    print(rhos)

    np.save("thesis_vectors{0}/mix_kw_{1}_{2}_{3}_{4}_{5}_{6}_power.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta, varx), powers)
    np.save("thesis_vectors{0}/mix_kw_{1}_{2}_{3}_{4}_{5}_{6}_pval.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta, varx), pvals)
    np.save("thesis_vectors{0}/mix_kw_{1}_{2}_{3}_{4}_{5}_{6}_power_kw.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta, varx), powers_kw)
    np.save("thesis_vectors{0}/mix_kw_{1}_{2}_{3}_{4}_{5}_{6}_pval_kw.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta, varx), pvals_kw)
    np.save("thesis_vectors{0}/mix_kw_{1}_{2}_{3}_{4}_{5}_{6}_rhos.npy".format(num_trials, slope1, slope2, frac, sigma_e, Delta, varx), rhos)

    powers1 = np.array(powers[0])
    powers2 = np.array(powers[1])
    powers3 = np.array(powers[2])
    powers4 = np.array(powers[3])
    powers1_kw = np.array(powers_kw[0])
    powers2_kw = np.array(powers_kw[1])
    powers3_kw = np.array(powers_kw[2])
    powers4_kw = np.array(powers_kw[3])
    plt.plot(ns, powers1, label=r"F-stat $\rho = (0.5)^2/2$")
    plt.plot(ns, powers2, label=r"F-stat $\rho = (1)^2/2$")
    plt.plot(ns, powers3, label=r"F-stat $\rho = (5)^2/2$")
    plt.plot(ns, powers4, label=r"F-stat non-private")
    plt.plot(ns, powers1_kw, label=r"KW $\rho = (0.5)^2/2$")
    plt.plot(ns, powers2_kw, label=r"KW $\rho = (1)^2/2$")
    plt.plot(ns, powers3_kw, label=r"KW $\rho = (5)^2/2$")
    plt.plot(ns, powers4_kw, label=r"KW non-private")
    plt.xlabel(r"$n$")
    plt.ylabel(r"Power")
    plt.legend()
    plt.savefig("thesis_images{0}/mix_kw_{1}_{2}_{3}_{4}_{5}_{6}_power.png".format(num_trials, slope1, slope2, frac, sigma_e, Delta, varx))
    plt.gcf().clear()

    pvals1 = np.array(pvals[0])
    pvals2 = np.array(pvals[1])
    pvals3 = np.array(pvals[2])
    pvals4 = np.array(pvals[3])
    pvals1_kw = np.array(pvals_kw[0])
    pvals2_kw = np.array(pvals_kw[1])
    pvals3_kw = np.array(pvals_kw[2])
    pvals4_kw = np.array(pvals_kw[3])
    plt.plot(ns, pvals1, label=r"F-stat $\rho = (0.5)^2/2$")
    plt.plot(ns, pvals2, label=r"F-stat $\rho = (1)^2/2$")
    plt.plot(ns, pvals3, label=r"F-stat $\rho = (5)^2/2$")
    plt.plot(ns, pvals4, label=r"F-stat non-private")
    plt.plot(ns, pvals1_kw, label=r"KW $\rho = (0.5)^2/2$")
    plt.plot(ns, pvals2_kw, label=r"KW $\rho = (1)^2/2$")
    plt.plot(ns, pvals3_kw, label=r"KW $\rho = (5)^2/2$")
    plt.plot(ns, pvals4_kw, label=r"KW non-private")
    plt.xlabel(r"$n$")
    plt.ylabel(r"p-value")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("thesis_images{0}/mix_kw_{1}_{2}_{3}_{4}_{5}_{6}_pval.png".format(num_trials, slope1, slope2, frac, sigma_e, Delta, varx))
    plt.gcf().clear()

if __name__ == "__main__":
    varx = 1
    barx = 0.5

    ### Linear
    sigma_e = 0.35
    Delta = 2
    
    ## vary distribution of 

    # uniform
    low = 0
    high = 1

    gen_data_n = lambda n, sigma_e, slope : gen_data_lin_norm(n, varx, barx, sigma_e**2, slope, 0.2)
    gen_data_u = lambda n, sigma_e, slope : gen_data_lin_unif(n, low, high, sigma_e**2, slope, 0.2)
    scale = 0.5
    gen_data_e = lambda n, sigma_e, slope : gen_data_lin_exp(n, scale, sigma_e**2, slope, 0.2)
    tester_n = lambda ym, b1, b2, x2m, s2, xm, t, tp, n, rho, Delta, alpha, sigma_e, stat : test.mc_lin_norm(ym, b1, b2, x2m, s2, xm, t, tp, n, rho, Delta, alpha, sigma_e, stat)
    tester_n_CI = lambda x, y, n, rho, alpha : test.mc_lin_norm_via_CI(x, y, n, rho, alpha)
    tester_n_Tulap = lambda x, y, n, rho, alpha : test.mc_lin_norm_via_Tulap(x, y, n, rho, alpha)

    #linear_CI(1, 0.35, Delta, gen_data_e, tester_n, tester_n_CI, "exp", varx, barx)
    #linear_CI(1, 0.35, Delta, gen_data_u, tester_n, tester_n_CI, "unif", varx, barx)
    #linear_CI(0, 0.35, Delta, gen_data_n, tester_n, tester_n_CI, "norm0", varx, barx)
    linear_Tulap(1, 0.35, Delta, gen_data_n, tester_n, tester_n_Tulap, "norm", varx, barx)

    '''
    ## vary slopes
    for slope in [1, 2]:
        for sigma_e in [0.35, 1, 3, 0.01]:
            linear(slope, sigma_e, Delta, gen_data_n, tester_n, "norm", varx, barx)

    ## vary slopes
    for slope in [0.1, 0.5]:
        for sigma_e in [0.35, 1, 3, 0.01]:
            linear(slope, sigma_e, Delta, gen_data_n, tester_n, "norm", varx, barx)

    ## vary distribution of x
    slope = 1
    sigma_e = 0.35
    Delta = 2
    # uniform
    low = 0
    high = 1
    gen_data_u = lambda n, sigma_e, slope : gen_data_lin_unif(n, low, high, sigma_e**2, slope, 0.2)
    # exponential
    scale = 0.5
    gen_data_e = lambda n, sigma_e, slope : gen_data_lin_exp(n, scale, sigma_e**2, slope, 0.2)
    # uniformly generated X with gaussian tester
    linear(slope, sigma_e, Delta, gen_data_u, tester_n, "unif_norm", varx, barx)
    # exponentially generated X with gaussian tester
    linear(slope, sigma_e, Delta, gen_data_e, tester_n, "exp_norm", varx, barx)
    '''
    ## vary slopes for CI method
    '''
    '''
    '''
    Delta = 3
    for (slope1, slope2) in [(-1, 1)]:
        for sigma_e in [0.35, 1, 0.01]:
            for frac in [1.0/2, 1.0/4, 1.0/8]:
                mixture(slope1, slope2, frac, sigma_e, Delta, varx, barx)
    Delta = 3
    for (slope1, slope2) in [(-0.1, 0.1)]:
        for sigma_e in [0.35, 1, 0.01]:
            for frac in [1.0/2, 1.0/4, 1.0/8]:
                mixture(slope1, slope2, frac, sigma_e, Delta, varx, barx)
    Delta = 3
    for (slope1, slope2) in [(-1.5, 1)]:
        for sigma_e in [0.35, 1, 0.01]:
            for frac in [1.0/2, 1.0/4, 1.0/8]:
                mixture(slope1, slope2, frac, sigma_e, Delta, varx, barx)
    Delta = 3
    for (slope1, slope2) in [(-1.5, 1.5)]:
        for sigma_e in [0.35, 1, 0.01]:
            for frac in [1.0/2, 1.0/4, 1.0/8]:
                mixture(slope1, slope2, frac, sigma_e, Delta, varx, barx)
    for (slope1, slope2) in [(-1, 1)]:
        mixture_kw(slope1, slope2, 0.5, 1, 3, 5, 0.1, barx)
        mixture_kw(slope1, slope2, 0.5, 1, 3, 5, 1, barx)
    for (slope1, slope2) in [(-1.5, 1.5)]:
        mixture_kw(slope1, slope2, 0.5, 1, 3, 5, 0.1, barx)
        mixture_kw(slope1, slope2, 0.5, 1, 3, 5, 1, barx)
    for (slope1, slope2) in [(-1, 1.5)]:
        mixture_kw(slope1, slope2, 0.5, 1, 3, 5, 0.1, barx)
        mixture_kw(slope1, slope2, 0.5, 1, 3, 5, 1, barx)
    '''

