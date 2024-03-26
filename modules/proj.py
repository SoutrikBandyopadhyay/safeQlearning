#!/usr/bin/env python3


import numpy as np


def proj_vec(param, param_dot, bound):
    """
    This is the projection operator obtained from the seminal paper.
        @article{lavretsky2011Arxiv,
        title={Projection operator in adaptive systems},
        author={Lavretsky, Eugene and Gibson, Travis E},
        journal={arXiv preprint arXiv:1112.4232},
        year={2011}
    }
    """
    theta = param
    y = param_dot
    f = np.matmul(theta.T, theta) - bound
    gradF = 2 * theta
    yggf = np.matmul(y.T, gradF)

    if f > 0 and yggf > 0:
        num = np.matmul(gradF, gradF.T)
        den = np.matmul(gradF.T, gradF)
        term2 = (np.matmul(num, y) / den) * f
        temp = y - term2
    else:
        temp = y

    return temp


def proj(parameterValue, parameterDerivative, gamma, bound):
    """
    This is the projection operator obtained from the seminal paper.
        @article{lavretsky2011Arxiv,
        title={Projection operator in adaptive systems},
        author={Lavretsky, Eugene and Gibson, Travis E},
        journal={arXiv preprint arXiv:1112.4232},
        year={2011}
    }
    """
    ans = []
    for i in range(parameterValue.shape[1]):
        theta = parameterValue[:, i].reshape(-1, 1)
        y = parameterDerivative[:, i].reshape(-1, 1)
        f = np.matmul(theta.T, theta) - bound
        gradF = 2 * theta

        yggf = gamma * np.matmul(y.T, gradF)

        if f > 0 and yggf > 0:
            num = np.matmul(gradF, gradF.T)
            den = gamma * np.matmul(gradF.T, gradF)
            term2 = gamma * (np.matmul(num, y) / den) * gamma * f
            temp = gamma * y - term2
        else:
            temp = gamma * y

        ans.append(temp.reshape(-1))

    return np.array(ans).T
