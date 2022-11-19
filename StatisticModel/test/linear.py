from abess import ConvexSparseSolver, make_glm_data
import numpy as np
import statistic_model_pybind

np.random.seed(1)
n = 30
p = 5
k = 3
family = "gaussian"

data = make_glm_data(family=family, n=n, p=p, k=k)

model = ConvexSparseSolver(model_size=p, support_size=k)

model.set_data(statistic_model_pybind.RegressionData(data.x,data.y))

model.set_model_user_defined(
    loss=statistic_model_pybind.linear_loss_no_intercept, 
    gradient=statistic_model_pybind.linear_gradient_no_intercept, 
    hessian=statistic_model_pybind.linear_hessian_no_intercept
)

model.set_log(0,6)

model.fit()

print(model.coef_)
print(data.coef_)
