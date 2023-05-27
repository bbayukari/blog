import numpy as np
import cvxpy as cp
import statistic_model_pybind

def logistic_loss_no_intercept(para, data):
    return statistic_model_pybind.logistic_loss_no_intercept(para, data)

def logistic_grad_no_intercept(para, data):
    return statistic_model_pybind.logistic_gradient_no_intercept(para, data)
    
def logistic_cvxpy_no_intercept(para, data):
    Xbeta = data.x @ para
    return cp.sum(
        cp.logistic(Xbeta) - cp.multiply(data.y, Xbeta)
    )   
