#include <random>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>

#include "autodiff/forward/dual.hpp"
//#include "autodiff/forward/dual/eigen.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;
using Eigen::Matrix;
namespace py = pybind11;
using autodiff::dual;
using autodiff::dual2nd;

//**********************************************************************************
// util func
//**********************************************************************************
MatrixXd slice(MatrixXd const &mat, VectorXi const &ind, int axis = 1) {
    MatrixXd A;
    if (axis == 0) {
        A.resize(ind.size(), mat.cols());
        if (ind.size() != 0) {
            for (Eigen::Index i = 0; i < ind.size(); i++) {
                A.row(i) = mat.row(ind(i));
            }
        }
    } else {
        A.resize(mat.rows(), ind.size());
        if (ind.size() != 0) {
            for (Eigen::Index i = 0; i < ind.size(); i++) {
                A.col(i) = mat.col(ind(i));
            }
        }
    }
    return A;
}

//**********************************************************************************
// regression data structure
//**********************************************************************************
struct RegressionData {
    MatrixXd x;
    VectorXd y;
    RegressionData(MatrixXd x, VectorXd y) : x(x), y(y){
        if (x.rows() != y.rows()) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
    }
};

//**********************************************************************************
// linear model 
//**********************************************************************************
double linear_loss_no_intercept(VectorXd const& para, VectorXd const& intercept, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    std::cout << "loss" << (data->x * para - data->y).squaredNorm() << std::endl;
    return (data->x * para - data->y).squaredNorm();  // compute the loss
}
VectorXd linear_gradient_no_intercept(VectorXd const& para, VectorXd const& intercept, py::object const& ex_data, VectorXi const& compute_para_index) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    std::cout << "gradient" << 2 * slice(data->x,compute_para_index).transpose() * (data->x * para - data->y) << std::endl;
    return 2 * slice(data->x,compute_para_index).transpose() * (data->x * para - data->y);  // compute the gradient
}
MatrixXd linear_hessian_no_intercept(VectorXd const& para, VectorXd const& intercept, py::object const& ex_data, VectorXi const& compute_para_index) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    MatrixXd X_compute = slice(data->x,compute_para_index);
    std::cout << "hessian" << X_compute.transpose() * X_compute << std::endl;
    return 2 * X_compute.transpose() * X_compute;  // compute the hessian
}

//**********************************************************************************
// logistic model
//**********************************************************************************
double logistic_loss_no_intercept(VectorXd const& para, VectorXd const& intercept, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta = (data->x * para).array();
    return ((xbeta.exp()+1.0).log() - (data->y).array()*xbeta).sum();
}

VectorXd logistic_gradient_no_intercept(VectorXd const& para, VectorXd const& intercept, py::object const& ex_data, VectorXi const& compute_para_index) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta_exp = (data->x * para).array().exp();
    return slice(data->x,compute_para_index).transpose() * (xbeta_exp / (xbeta_exp + 1.0) - (data->y).array()).matrix();
}

MatrixXd logistic_hessian_no_intercept(VectorXd const& para, VectorXd const& intercept, py::object const& ex_data, VectorXi const& compute_para_index) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta_exp = (data->x * para).array().exp();
    MatrixXd X_compute = slice(data->x,compute_para_index);
    return X_compute.transpose() * (1.0 / (xbeta_exp + 1.0 / xbeta_exp + 2.0) * X_compute.array()).matrix();
}

//**********************************************************************************
// Ising model data structure and generator
//**********************************************************************************
Eigen::MatrixXd comp_conf(int num_conf, int p){
  Eigen::MatrixXi conf = Eigen::MatrixXi::Zero(num_conf, p);
  Eigen::VectorXi num = Eigen::VectorXi::LinSpaced(num_conf, 0, num_conf - 1);

  for (int i = p - 1; i >= 0; i--){
    conf.col(i) = num - num / 2 * 2;
    num /= 2;
  }
  conf = conf.array() * 2 - 1;
  return conf.cast<double>();
}

// n is the number of samples
Eigen::MatrixXd sample_by_conf(int n, Eigen::MatrixXd theta, int seed) {
  int p = theta.rows();
  int num_conf = pow(2, p);
  
  Eigen::MatrixXd table = comp_conf(num_conf, p);
  Eigen::VectorXd weight(num_conf);
  
  Eigen::VectorXd vec_diag = theta.diagonal();
  Eigen::MatrixXd theta_diag = vec_diag.asDiagonal();
  Eigen::MatrixXd theta_off = theta - theta_diag;
  
  for (int num = 0; num < num_conf; num++) {
    Eigen::VectorXd conf = table.row(num);
    weight(num) = 0.5 * (double) (conf.transpose() * theta_off * conf) + (double) (vec_diag.transpose() * conf);
  }
  weight = weight.array().exp();
  
  std::vector<double> w;
  w.resize(weight.size());
  Eigen::VectorXd::Map(&w[0], weight.size()) = weight;
  
  // int sd = (((long long int)time(0)) * 2718) % 314159265;
  // Rcout << "Seed: "<< sd << endl;
  // std::default_random_engine generator(seed);  // implementation-defined
  // std::default_random_engine generator(1);

  std::mt19937_64 generator;                      // 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000
  generator.seed(seed);
  std::discrete_distribution<int> distribution(std::begin(w), std::end(w));
  
  Eigen::VectorXd freq = Eigen::VectorXd::Zero(num_conf);
  
  for (int i = 0; i < n; i++) {
    int num = distribution(generator);
    freq(num)++;
  }
  
  Eigen::MatrixXd data(num_conf, p + 1);
  data.col(0) = freq;
  // replace -1 with 0 in table
  data.rightCols(p) = table.array() * 0.5 + 0.5;
  return data;
}

struct IsingData{
    MatrixXd table;
    VectorXd freq;
    const int p;
    IsingData(MatrixXd X): p(X.cols()-1) {
        std::vector<int> index;
        for(Eigen::Index i = 0; i < X.rows(); i++){
            if(X(i,0) > 0.5){
                index.push_back(i);
            }
        }
        freq.resize(index.size());
        table.resize(index.size(),p);
        for(Eigen::Index i = 0; i < index.size(); i++){
            freq(i) = X(index[i],0);
            table.row(i) = X.row(index[i]).tail(p);
        }
    }
};
//**********************************************************************************
// Ising model loss
//**********************************************************************************
template <class T>
T ising_model(Matrix<T, -1, 1> const& para, Matrix<T, -1, 1> const& intercept, py::object const& ex_data) {
    IsingData* data = ex_data.cast<IsingData*>();
    T loss = T(0.0);
    
    for(int i = 0; i < data->table.rows(); i++){
        int idx = 0;
        for(int s = 0; s < data->p; s++){
            loss -= data->freq(i) * data->table(i,s) * intercept(s);
            for(int t = s+1; t < data->p; t++){
                loss -= 2 * data->freq(i) * data->table(i,s) * data->table(i,t) * para(idx++);
            }
        }
        for(int s = 0; s < data->p; s++){
            T tmp = intercept(s);
            for(int t = 0; t < data->p; t++){
                if(t > s)
                    tmp += para((2*data->p-s)*(s+1)/2+t-s-1-data->p) * data->table(i,t);
                else if(t < s)
                    tmp += para((2*data->p-t)*(t+1)/2+s-t-1-data->p) * data->table(i,t);
            }
            loss += data->freq(i) * log(1+exp(tmp));
        }
    }
    return loss;
}

//**********************************************************************************
// pybind11 module
//**********************************************************************************
PYBIND11_MODULE(statistic_model_pybind,m) {
    // export the data structure CustomData and its constructor
    pybind11::class_<RegressionData>(m, "RegressionData").def(py::init<MatrixXd, VectorXd>());
    // linear model
    m.def("linear_loss_no_intercept", &linear_loss_no_intercept);
    m.def("linear_gradient_no_intercept", &linear_gradient_no_intercept);
    m.def("linear_hessian_no_intercept", &linear_hessian_no_intercept);
    // logistic model
    m.def("logistic_loss_no_intercept", &logistic_loss_no_intercept);
    m.def("logistic_gradient_no_intercept", &logistic_gradient_no_intercept);
    m.def("logistic_hessian_no_intercept", &logistic_hessian_no_intercept);
    // ising model
    py::class_<IsingData>(m, "IsingData").def(py::init<MatrixXd>());
    m.def("ising_model",
          py::overload_cast<Matrix<double, -1, 1> const&, Matrix<double, -1, 1> const&, py::object const&>(
              &ising_model<double>));
    m.def("ising_model",
          py::overload_cast<Matrix<dual, -1, 1> const&, Matrix<dual, -1, 1> const&, py::object const&>(
              &ising_model<dual>));
    m.def(
        "ising_model",
        py::overload_cast<Matrix<dual2nd, -1, 1> const&, Matrix<dual2nd, -1, 1> const&, py::object const&>(
            &ising_model<dual2nd>));
}
