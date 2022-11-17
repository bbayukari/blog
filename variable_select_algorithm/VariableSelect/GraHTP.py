import numpy as np
import nlopt

def GraHTP(
    loss,
    grad,
    dim,
    support_size,
    data=None,
    fast=False,
    final_support_size=-1,
    x_init=None,
    step_size=0.005,
    max_iter=100,
):
    """GraHTP algorithm
    Args:
        loss: function (x, data, active_index) -> loss_value
        grad: function (x, data, active_index, compute_index) -> gradient_vector
            x: array with the shape of (dim,)
            data: dictionary, data for loss and grad
            active_index: int array, the index of nonzore features, default is None which means it's np.arange(dim)
            compute_index: int array, the index of features which gradient is computed
            gradient_vector.shape = compute_index.shape, default is None which means it's same with active_index
        dim: int, dimension of the model which is the length of x
        support_size: the number of selected features for algorithm
        data: dictionary, data for loss and grad
        fast: bool, whether to use FGraHTP algorithm or not
        final_support_size: int, must less than support_size, algorithm will select support_size features firstly,
            then preserve the top final_support_size entries of the output as the final estimation, default is -1 which means it's same with support_size.
        x_init: the initial value of the estimator, default is None which means it's np.zeros(dim).
    Returns:
        estimator: array with the shape of (dim,) which contains k nonzero entries
    """
    if x_init is None:
        x_init = np.zeros(dim)
    
    if final_support_size < 0:
        final_support_size = support_size
    
    # init
    x_old = x_init
    support_old = np.argpartition(np.abs(x_old), -support_size)[-support_size:] # the index of support_size largest entries

    for iter in range(max_iter):
        # S1: gradient descent
        x_bias = x_old - step_size * grad(x_old, data)
        # S2: Gradient Hard Thresholding
        support_new = np.argpartition(np.abs(x_bias), -support_size)[-support_size:]
        # S3: debise
        if fast:
            x_new = np.zeros(dim)
            x_new[support_new] = x_bias[support_new]
        else:
            try:
                def opt_f(x, gradient):
                    x_full = np.zeros(dim)
                    x_full[support_new] = x
                    if gradient.size > 0:
                        gradient[:] = grad(x_full, data, support_new)
                    return loss(x_full, data, support_new)    

                opt = nlopt.opt(nlopt.LD_SLSQP, support_size)
                opt.set_min_objective(opt_f)
                opt.set_ftol_rel(0.001)
                x_new = np.zeros(dim)
                x_new[support_new] = opt.optimize(x_bias[support_new])
            except RuntimeError:
                raise
        # terminating condition
        if np.all(set(support_old) == set(support_new)):
            break
        x_old = x_new
        support_old = support_new

    final_support = np.argpartition(np.abs(x_new), -final_support_size)[-final_support_size:]
    final_estimator = np.zeros(dim)
    final_estimator[final_support] = x_new[final_support]

    return final_estimator

def GraHTP_cv(loss_fn, grad_fn, dim, support_size, data):
    step_size_cv = [0.0001, 0.0005, 0.05, 0.1] + [(s + 1) / 1000 for s in range(10)]

    best_estimator = np.zeros(dim)
    min_loss = loss_fn(best_estimator, data)
    best_step_size = 0.0
    fail_times = 0
    for step_size in step_size_cv:
        try:
            x = GraHTP(loss_fn, grad_fn, dim, support_size, step_size=step_size, data=data)
            loss = loss_fn(x, data)
            if loss < min_loss:
                min_loss = loss
                best_estimator = x
                best_step_size = step_size
        except RuntimeError:
            fail_times += 1
            if fail_times > 4:
                raise

    return best_estimator, best_step_size
