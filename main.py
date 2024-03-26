#!/usr/bin/env python3


import datetime
import os

import control
import matplotlib as mpl
import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.io import loadmat, savemat

from modules.barriers import NormBarrier
from modules.costfunctions import QuadraticCost
from modules.safe_ql import SafeQL

sns.set_style("whitegrid")
sns.color_palette("colorblind")
#
# modify global setting
font_size = 14
legend_font = 11
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["lines.linewidth"] = 4
mpl.rcParams["axes.titlesize"] = font_size
mpl.rcParams["axes.labelsize"] = font_size
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["lines.markersize"] = 10
mpl.rcParams["xtick.labelsize"] = font_size
mpl.rcParams["ytick.labelsize"] = font_size
mpl.rcParams["legend.fontsize"] = legend_font
mpl.rcParams["mathtext.fontset"] = "cm"

if __name__ == "__main__":
    np.random.seed(786)
    state_dim = 2
    action_dim = 1
    A = np.array(
        [
            [0, 1],
            [1.6, 2.8],
        ]
    )

    B = np.array(
        [
            [0],
            [1],
        ]
    )

    M = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    R = 0.1 * np.eye(action_dim)

    init_x = np.array([[1, 1]]).T

    barrier = NormBarrier(state_dim, upper_lim=1.5)
    cost = QuadraticCost(M, R)

    K, P, E = control.lqr(A, B, 2 * M, 2 * R)

    idealWa = -K.T

    idealQxx = P + M + np.matmul(P, A) + np.matmul(A.T, P)
    idealQxu = np.matmul(P, B)
    idealQux = np.matmul(B.T, P)
    idealQuu = R

    idealQ = np.block(
        [
            [idealQxx, idealQxu],
            [idealQux, idealQuu],
        ]
    )

    x = np.array([[1], [1]])

    learner = SafeQL(A, B, barrier, cost, idealK=idealWa, idealQ=idealQ)

    idealWc = 0.5 * learner.vech(idealQ)

    learner.set_init_condition(x=init_x)

    init_wa = np.array([[-1.0, 1.0]]).T
    init_wc = idealWc + 2 * np.ones_like(idealWc)

    learner.lamb = 0.2
    learner.actor_gain = 1
    learner.critic_gain = 150

    data = learner.simulate(init_wa, init_wc)

    x = data["sol"][:, :-1]

    data_export = {
        "T": data["T"],
        "cost": data["sol"][:, -1],
        "x": x,
        "u": data["u"],
        "wa_hat": data["wa_hat"],
        "wc_hat": data["wc_hat"],
    }
    instant = datetime.datetime.now()
    filename = instant.strftime(f"%Y%m%d_Data_%H%M%S")

    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    finally:
        pass

    savemat(
        f"data/{filename}.mat",
        data_export,
        do_compression=True,
    )

    print(f"cost = {data['sol'][-1, -1]}"),

    fig = plt.figure(figsize=(8, 6), layout="constrained")
    spec = fig.add_gridspec(3, 2)
    ax00 = fig.add_subplot(spec[0, :])
    ax10 = fig.add_subplot(spec[1, :])
    ax02 = fig.add_subplot(spec[2, :])

    ax00.plot(data["T"].T, x[:, 0], label="$x(1)$")
    ax00.plot(data["T"].T, x[:, 1], label="$x(2)$")

    def norm(x):
        """Compute norm of 2d matrix."""
        return np.sqrt(np.sum(np.power(x, 2), axis=1))

    wa_hat = data["wa_hat"]

    ax10.plot(data["T"].T, wa_hat[:, 0], label=r"$\hat{W}_a(1)$")
    ax10.plot(data["T"].T, wa_hat[:, 1], label=r"$\hat{W}_a(2)$")
    ax10.plot(
        data["T"].T,
        idealWa[0, 0] * np.ones_like(data["T"].T),
        linestyle="-.",
        label=r"$W_a(1)$",
    )
    ax10.plot(
        data["T"].T,
        idealWa[1, 0] * np.ones_like(data["T"].T),
        linestyle="-.",
        label=r"$W_a(2)$",
    )
    ax02.plot(data["T"].T, norm(x), label="Safe Q-Learning")

    ax02.plot(
        data["T"].T,
        1.5 * np.ones_like(data["T"].T),
        color="red",
        linestyle="-.",
        label="Constraint",
    )

    ax00.legend(loc=5)
    ax10.legend(loc=5)

    ax00.set_xlim([-1, 30])
    ax10.set_xlim([-1, 30])
    ax02.set_xlim([-1, 30])

    ax00.set_xlabel("Time(in sec)\n(a)")
    ax10.set_xlabel("Time(in sec)\n(b)")
    ax02.set_xlabel("Time(in sec)\n(c)")

    ax00.set_ylabel(r"$x$")
    ax10.set_ylabel(r"$\hat{W}_a$")
    ax02.set_ylabel(r"$\|x\|$")

    ax02.legend()

    fig.savefig("finalPlot.png")

    plt.show()
