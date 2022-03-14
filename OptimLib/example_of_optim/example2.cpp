#define OPTIM_ENABLE_EIGEN_WRAPPERS // it means that we'll use the eigen lib to compute
#include "optim.hpp" 
#include <autodiff/forward/real.hpp> // these two lines make auto differential available
#include <autodiff/forward/real/eigen.hpp>
// this function is object function, these special parameter types are required for automatic differentiation
autodiff::real opt_fnd(const autodiff::ArrayXreal& x){
    return x.cwiseProduct(x).sum(); // L2 norm of vector
}
// this function is to be provided to the algorithm function later
// because it can compute both function value and its differential
double opt_fn(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data) {
    autodiff::real u; // u is to be function value
    autodiff::ArrayXreal xd = x.eval(); // this is an assignment statement
    if (grad_out) { // if grad_out != NULL, we should compute differential, otherwise we needn't
        Eigen::VectorXd grad_tmp = autodiff::gradient(opt_fnd, autodiff::wrt(xd), autodiff::at(xd), u);
        *grad_out = grad_tmp; // this is the differntial 
    } else {
        u = opt_fnd(xd); // only compute function value
    }
    return u.val(); // return function value
}

int main() {
    Eigen::VectorXd x(5); // this is initial value
    x << 1, 2, 3, 4, 5;

    optim::bfgs(x, opt_fn, nullptr); // this is Quasi-Newton Method

    std::cout << "solution: x = \n" << x << std::endl;

    return 0;
}
