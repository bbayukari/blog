import numpy as np
import nlopt


def GraSP(
    loss_fn,
    dim,
    support_size,
    data,
    grad_fn=None,
    max_iter=100
):
    """GraSP algorithm
    Args:
         loss_fn: function (x, data) -> loss_value
            x: array with the shape of (dim,)
            data: dictionary, data for loss and grad
        grad_fn: function (x, data, compute_para_index) -> gradient_vector 
            x, data: same as loss_fn
            compute_para_index: array which contains the index of parameters need to compute gradient
            default: None, algorithm will compute gradient by jax, in this case loss_fn must be coded by jax.
        dim: int, dimension of the model which is the length of x
        support_size: the number of selected features for algorithm
        data: dictionary, data for loss and grad
    Returns:
        estimator: array with the shape of (dim,) which contains k nonzero entries
    """
    if grad_fn is None:
        from jax import jacfwd
        import jax.numpy as jnp

        loss_fn_jax = loss_fn
        loss_fn = lambda x, data: loss_fn_jax(x, data).item()

        def func_(para_compute, data, para, index):
            para_complete = para.at[index].set(para_compute)
            return loss_fn_jax(para_complete, data)

        def grad_fn(para, data, compute_para_index):
            para_j = jnp.array(para)
            para_compute_j = jnp.array(para[compute_para_index])
            return np.array(
                jacfwd(func_)( ## forward mode automatic differentiation is faster than reverse mode
                    para_compute_j, data, para_j, compute_para_index
                )
            )
    # init
    x_old = np.zeros(dim)
 
    for iter in range(max_iter):
        # compute local gradient 
        z = grad_fn(x_old, data, np.arange(dim))

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
                    gradient[:] = grad_fn(x_full, data, support_new)
                return loss_fn(x_full, data)    

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
