import numpy as np
import nlopt


def GraHTP(
    loss_fn,
    dim,
    support_size,
    data,
    grad_fn=None,
    fast=False,
    final_support_size=-1,
    x_init=None,
    step_size=0.005,
    max_iter=100,
):
    """GraHTP algorithm
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
        fast: bool, whether to use FGraHTP algorithm or not
        final_support_size: int, must less than support_size, algorithm will select support_size features firstly,
            then preserve the top final_support_size entries of the output as the final estimation, default is -1 which means it's same with support_size.
        x_init: the initial value of the estimator, default is None which means it's np.zeros(dim).
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
                jacfwd(
                    func_
                )(  ## forward mode automatic differentiation is faster than reverse mode
                    para_compute_j, data, para_j, compute_para_index
                )
            )

    if x_init is None:
        x_init = np.zeros(dim)

    if final_support_size < 0:
        final_support_size = support_size

    # init
    x_old = x_init
    support_old = np.argpartition(np.abs(x_old), -support_size)[
        -support_size:
    ]  # the index of support_size largest entries

    for iter in range(max_iter):
        # S1: gradient descent
        x_bias = x_old - step_size * grad_fn(x_old, data, np.arange(dim))
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
                        gradient[:] = grad_fn(x_full, data, support_new)
                    return loss_fn(x_full, data)

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

    final_support = np.argpartition(np.abs(x_new), -final_support_size)[
        -final_support_size:
    ]
    final_estimator = np.zeros(dim)
    final_estimator[final_support] = x_new[final_support]

    return final_estimator


def GraHTP_cv(loss_fn, dim, support_size, data, grad_fn=None):
    step_size_cv = np.logspace(-4, -2, 10)

    best_estimator = np.zeros(dim)
    min_loss = loss_fn(best_estimator, data)
    best_step_size = 0.0
    fail_times = 0
    for step_size in step_size_cv:
        try:
            x = GraHTP(
                loss_fn=loss_fn,
                dim=dim,
                support_size=support_size,
                data=data,
                grad_fn=grad_fn,
                step_size=step_size,
            )
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
