import numpy as np
import cvxpy as cp

def linear_loss_no_intercept(para, data):
    return np.sum(np.square(data.y - data.x @ para))

def linear_grad_no_intercept(para, data, compute_para_index):
    return -2 * data.x[:,compute_para_index].T @ (data.y - data.x @ para)

def linear_cvxpy_no_intercept(para, data):
    return cp.sum_squares(data.y - data.x @ para)
