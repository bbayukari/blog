import numpy as np
import nlopt


def GraSP(
    loss,
    grad,
    dim,
    support_size,
    data=None,
    max_iter=100
):
    """GraSP algorithm
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
    Returns:
        estimator: array with the shape of (dim,) which contains k nonzero entries
    """

    # init
    x_old = np.zeros(dim)
 
    for iter in range(max_iter):
        # compute local gradient 
        z = grad(x_old, data)

        # identify directions
        if 2*support_size < dim:
            Omega = np.nonzero(z[np.argpartition(np.abs(z), -2*support_size)[-2*support_size:]])[0] # supp(z_2s)
        else:
            Omega = np.nonzero(z)[0] # supp(z)

        # merge supports
        support_new = np.unique(np.append(Omega, x_old.nonzero()[0])) 
        


        # minimize 
        try:
            def opt_f(x, gradient):
                x_full = np.zeros(dim)
                x_full[support_new] = x
                if gradient.size > 0:
                    gradient[:] = grad(x_full, data, support_new)
                return loss(x_full, data, support_new)    

            opt = nlopt.opt(nlopt.LD_SLSQP, support_new.size)
            opt.set_min_objective(opt_f)
            opt.set_ftol_rel(0.001)
            x_tem = np.zeros(dim)
            x_tem[support_new] = opt.optimize(x_old[support_new])
        except RuntimeError:
            raise
        
        # prune estimate
        x_supp = np.argpartition(np.abs(x_tem), -support_size)[-support_size:]
        x_new = np.zeros(dim)
        x_new[x_supp] = x_tem[x_supp]

        # update
        x_old = x_new
        support_old = support_new

        # terminating condition
        if np.all(set(support_old) == set(support_new)):
            break

    return x_new
