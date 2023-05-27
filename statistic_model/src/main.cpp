#include <random>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>

#include "autodiff/forward/dual.hpp"
//#include "autodiff/forward/dual/eigen.hpp" // not needed because ising_model() does not use Matrix multiplication of Eigen 

using Eigen::MatrixXd;
using Eigen::MatrixXi;
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
VectorXd slice(VectorXd const &v, VectorXi const &idx) {
    VectorXd v2(idx.size());
    for (int i = 0; i < idx.size(); i++) {
        v2(i) = v(idx(i));
    }
    return v2;
}

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
double linear_loss_no_intercept(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    return (data->x * para - data->y).squaredNorm();  // compute the loss
}
VectorXd linear_gradient_no_intercept(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    VectorXd result = 2 * data->x.transpose() * (data->x * para - data->y);
    return result;  // compute the gradient
}
MatrixXd linear_hessian_no_intercept(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    MatrixXd X_compute = data->x;
    MatrixXd result = 2 * X_compute.transpose() * X_compute;
    return result;  // compute the hessian
}

//**********************************************************************************
// logistic model
//**********************************************************************************
double logistic_loss_no_intercept(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta = (data->x * para).array();
    return ((xbeta.exp()+1.0).log() - (data->y).array()*xbeta).sum();
}

VectorXd logistic_gradient_no_intercept(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta_exp = (data->x * para).array().exp();
    return data->x.transpose() * (xbeta_exp / (xbeta_exp + 1.0) - (data->y).array()).matrix();
}

MatrixXd logistic_hessian_no_intercept(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta_exp = (data->x * para).array().exp();
    MatrixXd X_compute = data->x;
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
Eigen::MatrixXd sample_by_conf(int sample_size, Eigen::MatrixXd theta, int seed) {
  int p = theta.rows();
  int num_conf = pow(2, p);
  
  Eigen::MatrixXd table = comp_conf(num_conf, p);
  Eigen::VectorXd weight(num_conf);
  
  Eigen::VectorXd vec_diag = theta.diagonal();
  Eigen::MatrixXd theta_diag = vec_diag.asDiagonal();
  Eigen::MatrixXd theta_off = theta - theta_diag;
  
  for (int num = 0; num < num_conf; num++) {
    Eigen::VectorXd conf = table.row(num);
    weight(num) = 0.5 * (double) (conf.transpose() * theta_off * conf);
  }
  weight = weight.array().exp();
  
  std::vector<double> w;
  w.resize(weight.size());
  Eigen::VectorXd::Map(w.data(), weight.size()) = weight;
  
  // int sd = (((long long int)time(0)) * 2718) % 314159265;
  // Rcout << "Seed: "<< sd << endl;
  // std::default_random_engine generator(seed);  // implementation-defined
  // std::default_random_engine generator(1);

  std::mt19937_64 generator;                      // 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000
  generator.seed(seed);
  std::discrete_distribution<int> distribution(std::begin(w), std::end(w));
  
  Eigen::VectorXd freq = Eigen::VectorXd::Zero(num_conf);
  
  for (int i = 0; i < sample_size; i++) {
    freq(distribution(generator))++;
  }
  
  Eigen::MatrixXd data(num_conf, p + 1);
  data.col(0) = freq;
  data.rightCols(p) = table;
  return data;
}

struct IsingData{
    MatrixXd table;
    VectorXd freq;
    const int p;
    MatrixXi index_translator;
    IsingData(MatrixXd X): p(X.cols()-1) {
        table = X.rightCols(p);
        freq = X.col(0);

        index_translator.resize(p,p);
        int count = 0;
        for(Eigen::Index i = 0; i < p; i++){
            for(Eigen::Index j = i+1; j < p; j++){
                index_translator(i,j) = count;
                index_translator(j,i) = count;
                count++;
            }
        }
    }
};
//**********************************************************************************
// Ising model loss
//**********************************************************************************
template <class T>
T ising_model(Matrix<T, -1, 1> const& para, py::object const& ex_data) {
    IsingData* data = ex_data.cast<IsingData*>();
    T loss = T(0.0);

    for(int i = 0; i < data->table.rows(); i++){
        for(int k=0; k< data->p; k++){
            T tmp = T(0.0);
            for(int j = 0; j < data->p; j++){
                if (j == k) continue;
                tmp += data->table(i,k) * data->table(i,j) * para(data->index_translator(k,j));
            }
            loss += data->freq(i) * log(1+exp(-2 * tmp));
        }
    }
    return loss;
}

VectorXd ising_grad(VectorXd const& para, py::object const& ex_data) {
    IsingData* data = ex_data.cast<IsingData*>();
    VectorXd grad_para = VectorXd::Zero(para.size());

    for(int i = 0; i < data->table.rows(); i++){
        for(int k=0; k< data->p; k++){
            double tmp = 0.0;
            for(int j = 0; j < data->p; j++){
                if (j == k) continue;
                tmp += data->table(i,k) * data->table(i,j) * para(data->index_translator(k,j));
            }
            double exp_tmp = 2 * data->freq(i) * data->table(i,k)  / (1+exp(2 * tmp));
            for(int j = 0; j < data->p; j++){
                if (j == k) continue;
                grad_para(data->index_translator(k,j)) -= exp_tmp * data->table(i,j);
            }
        }
    }

    return grad_para;
}

MatrixXd ising_hess_diag(VectorXd const& para, py::object const& ex_data) {  
    static VectorXd hess_diag_para(para.size());
    static VectorXd para_last; // store the last para

    // if para is not changed, return the last hessian diagonal; otherwise, recompute the hessian diagonal
    if(para_last.size() != para.size() || (para - para_last).squaredNorm() > 1e-6){
        hess_diag_para.setZero();
        para_last = para;
        IsingData* data = ex_data.cast<IsingData*>();

        for(int i = 0; i < data->table.rows(); i++){
            for(int k=0; k< data->p; k++){
                double tmp = 0.0;
                for(int j = 0; j < data->p; j++){
                    if (j == k) continue;
                    tmp += data->table(i,k) * data->table(i,j) * para(data->index_translator(k,j));
                }
                double phi = 1.0 / (1.0 + exp(2 * tmp));
                double h = 4 * data->freq(i) * phi * (1 - phi);
                for(int j = 0; j < data->p; j++){
                    if (j == k) continue;
                    hess_diag_para(data->index_translator(k,j)) += h;
                }
            }
        }
    }

    return hess_diag_para.asDiagonal();
}

/*
template <class T>
T ising_model(Matrix<T, -1, 1> const& para, Matrix<T, -1, 1> const& intercept, py::object const& ex_data) noexcept{
    IsingData* data = ex_data.cast<IsingData*>();
    T loss = T(0.0);
    bool has_intercept = intercept.size() == data->p;
    
    for(int i = 0; i < data->table.rows(); i++){
        int idx = 0;
        for(int s = 0; s < data->p; s++){
            if(has_intercept){
                loss -= data->freq(i) * data->table(i,s) * intercept(s);
            }
            for(int t = s+1; t < data->p; t++){
                loss -= 2 * data->freq(i) * data->table(i,s) * data->table(i,t) * para(idx++);
            }
        }
        for(int s = 0; s < data->p; s++){
            T tmp = has_intercept ? intercept(s) : T(0.0);
            for(int t = 0; t < data->p; t++){
                if (t == s) continue;
                tmp += para(data->index_translator(s,t)) * data->table(i,t);
            }
            loss += data->freq(i) * log(1+exp(tmp));
        }
    }
    return loss;
}
VectorXd ising_grad_no_intercept(VectorXd const& para, py::object const& ex_data, VectorXi const& compute_para_index) {
    IsingData* data = ex_data.cast<IsingData*>();
    VectorXd grad_para = VectorXd::Zero(para.size());

    for(int i = 0; i < data->table.rows(); i++){
        int idx = 0;
        for(int s = 0; s < data->p; s++){
            for(int t = s+1; t < data->p; t++){
                grad_para(idx) -= 2 * data->freq(i) * data->table(i,s) * data->table(i,t);
                idx += 1;
            }
        }
        for(int s = 0; s < data->p; s++){
            double local_loss = 0.0;
            for(int t = 0; t < data->p; t++){
                if(t != s){
                    local_loss += para(data->index_translator(s,t)) * data->table(i,t);
                }
            }
            double local_grad_coef = data->freq(i) / (1+exp(-local_loss));
            for(int t = 0; t < data->p; t++){
                if(t != s){
                    grad_para(data->index_translator(s,t)) += local_grad_coef * data->table(i,t);
                }
            }
        }
    }

    VectorXd grad = VectorXd::Zero(compute_para_index.size());
    for(Eigen::Index i=0; i<compute_para_index.size(); i++){
        grad(i) = grad_para(compute_para_index(i));
    }
    return grad;
}

MatrixXd ising_hess_diag_no_intercept(VectorXd const& para, py::object const& ex_data, VectorXi const& compute_para_index) {
    if (compute_para_index.size() > 1){
        throw std::runtime_error("Hessian diagonal is only available for one parameter at a time.");
    }
    
    static VectorXd hess_diag_para(para.size());
    static VectorXd para_last; // store the last para

    // if para is not changed, return the last hessian diagonal; otherwise, recompute the hessian diagonal
    if(para_last.size() != para.size() || (para - para_last).squaredNorm() > 1e-6){
        hess_diag_para.setZero();
        para_last = para;
        IsingData* data = ex_data.cast<IsingData*>();

        for(int i = 0; i < data->table.rows(); i++){
            for(int s = 0; s < data->p; s++){
                double local_loss = 0.0;
                for(int t = 0; t < data->p; t++){
                    if(t != s){
                        local_loss += para(data->index_translator(s,t)) * data->table(i,t);
                    }
                }
                double logistic_coef = 1.0 / (1+exp(-local_loss));
                double local_hess_coef = data->freq(i) * logistic_coef * (1-logistic_coef);
                for(int t = 0; t < data->p; t++){
                    if(t != s){
                        hess_diag_para(data->index_translator(s,t)) += local_hess_coef * data->table(i,t);
                    }
                }
            }
        }
    }
   
    return hess_diag_para(compute_para_index(0)) * MatrixXd::Identity(1,1);
}
*/

//**********************************************************************************
// pybind11 module
//**********************************************************************************
PYBIND11_MODULE(statistic_model_pybind,m) {
    // data structure CustomData and its constructor
    pybind11::class_<RegressionData>(m, "RegressionData").def(py::init<MatrixXd, VectorXd>());
    // linear model
    m.def("linear_loss_no_intercept", &linear_loss_no_intercept);
    m.def("linear_gradient_no_intercept", &linear_gradient_no_intercept);
    m.def("linear_hessian_no_intercept", &linear_hessian_no_intercept);
    // logistic model
    m.def("logistic_loss_no_intercept", &logistic_loss_no_intercept);
    m.def("logistic_gradient_no_intercept", &logistic_gradient_no_intercept);
    m.def("logistic_hessian_no_intercept", &logistic_hessian_no_intercept);
    // Ising data constructor
    m.def("ising_generator", &sample_by_conf);
    // data structure IsingData and its constructor
    py::class_<IsingData>(m, "IsingData").def(py::init<MatrixXd>());
    // ising model
    m.def("ising_model",
          py::overload_cast<Matrix<double, -1, 1> const&, py::object const&>(
              &ising_model<double>));
    m.def("ising_model",
          py::overload_cast<Matrix<dual, -1, 1> const&, py::object const&>(
              &ising_model<dual>));
    m.def(
        "ising_model",
        py::overload_cast<Matrix<dual2nd, -1, 1> const&, py::object const&>(
            &ising_model<dual2nd>));
    // ising model explicit expression
    m.def("ising_loss", &ising_model<double>);
    m.def("ising_grad", &ising_grad);
    m.def("ising_hess_diag", &ising_hess_diag);
}
