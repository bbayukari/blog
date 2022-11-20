from math import exp, log
import numpy as np
import cvxpy as cp
import statistic_model_pybind

def logistic_loss_no_intercept(para, data):
    """
    compute loss of logistic function
    Args:
        para: array with the shape of (p,)
        data: class
            x: array with the shape of (n,p)
            y: array with the shape of (n,)
    Returns:
        value: 1'log(1 + exp(Xbeta)) - y'Xbeta
    
    Xbeta = data.x @ para
    return sum(
        [x if x > 100 else 0.0 if x < -100 else log(1 + exp(x)) for x in Xbeta]
    ) - np.dot(data.y, Xbeta)
    """
    return statistic_model_pybind.logistic_loss_no_intercept(para, np.zeros(0), data)

def logistic_grad_no_intercept(para, data, compute_para_index):
    """
    compute grad of logistic function
    Args:
        para: array with the shape of (p,)
        data: class
            x: array with the shape of (n,p)
            y: array with the shape of (n,)
        compute_para_index: array which contains the index of parameters need to compute gradient
    Returns:
        grad: X(1/(1+exp(-Xbeta)) - y on the index of compute_para_index
    
    tem = np.array([0.0 if x < -100 else 1.0 if x > 100 else 1 / (1 + exp(-x)) for x in data.x @ para])
    return data.x[:,compute_para_index].T @ (tem - data.y)
    """
    return statistic_model_pybind.logistic_gradient_no_intercept(para, np.zeros(0), data, compute_para_index)
    
def logistic_cvxpy_no_intercept(para, data):
    Xbeta = data.x @ para
    return cp.sum(
        cp.logistic(Xbeta) - cp.multiply(data.y, Xbeta)
    )   
