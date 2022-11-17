import cvxpy as cp
import numpy as np


def lasso(loss_cvxpy, dim, support_size, data=None):
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
    min_estimator = np.zeros(dim)
    min_loss = loss_cvxpy(min_estimator, data)

    for v in np.logspace(-2, 3, 50):
        lambd.value = v
        problem.solve()
        loss = loss_cvxpy(x.value, data)
        if loss < min_loss:
            min_loss = loss
            min_estimator = x.value
    
    return min_estimator
