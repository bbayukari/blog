#define OPTIM_ENABLE_EIGEN_WRAPPERS // it means that we'll use the eigen lib to compute
#include "optim.hpp" // it means we'll use this optim lib

inline // cpp grammar, it's not necessary
double sphere_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
{ // this function is object function which can compute both function value and its differential
  // and it is to be provided to the algorithm function later
    double obj_val = vals_inp.dot(vals_inp); // obj_val is to be function value

    if (grad_out) { // if grad_out != NULL, we should compute differential, otherwise we needn't
        *grad_out = 2.0*vals_inp; // this is the differntial 
    }

    return obj_val; // return function value
}

int main()
{
    const int test_dim = 5;

    Eigen::VectorXd x = Eigen::VectorXd::Ones(test_dim); // initial values (1,1,...,1)

    optim::algo_settings_t settings; // this is to set para for algorithm

    settings.print_level = 4; // it means that all information will be printed

    bool success = optim::gd(x, sphere_fn, nullptr, settings); // this is Gradient Descent Method

    if (success) {
        std::cout << "gd: sphere test completed successfully." << "\n";
    } else {
        std::cout << "gd: sphere test completed unsuccessfully." << "\n";
    }

    std::cout << "gd: solution to sphere test:\n" << x << std::endl;

    return 0;
}
