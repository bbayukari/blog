import cvxpy as cp
import numpy as np


def Lasso(loss_cvxpy, dim, support_size, data=None, tol=1, init_lambda=1.0, max_iter=100):
    """lasso algorithm
    Args:
        loss_cvxpy: cvxpy function (x, data) -> loss_value
            x: array with the shape of (dim,)
            data: dictionary, data for loss 
            
        dim: int, dimension of the model which is the length of x
        support_size: the number of selected features for algorithm
        data: dictionary, data for loss 
       
    Returns:
        estimator: array with the shape of (dim,) which contains support_size nonzero entries
    """
    def object_fn(x, lambd):
        return loss_cvxpy(x, data) + lambd * cp.norm1(x)

    x = cp.Variable(dim)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(object_fn(x, lambd)))

    lambd_lowwer = 0.0
    lambd.value = init_lambda

    for i in range(max_iter):
        problem.solve()
        estimator = x.value
        support_size_est = np.array(abs(estimator) > tol).sum() 

        if support_size_est > support_size:
            lambd_lowwer = lambd.value
            lambd.value = 2 * lambd.value
        elif support_size_est < support_size:
            lambd.value = (lambd_lowwer + lambd.value) / 2
        else:
            break
    
    estimator[abs(estimator) < tol] = 0.0
    return estimator, lambd.value
