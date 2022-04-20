#include <iostream>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <functional>

using namespace Eigen;
using namespace autodiff;
using namespace std;

template<typename T>
T test(Matrix<T,-1,1> const &para, MatrixXd &x){
    return (x*para).squaredNorm();
}

int main() {
    MatrixXd data = MatrixXd::Random(3,3);

    //函数求值
    VectorXd x(3);
    x << 1,15,10;
    double v = test<double>(x,data);
    cout << v << endl;

    // 函数求导，顺便求值
    VectorXdual x1 = x; // 注意阶数匹配
    dual vd1; // 不是double
    VectorXd g1 = gradient([&data](const VectorXdual& para){return test<dual>(para,data);}, wrt(x1), at(x1),vd1);
    double v1 = val(vd1);  // 必须显式转换
    cout << v1 << endl << g1 << endl;

    // 函数求Hessian，顺便求导，求值
    VectorXd y(4);
    y << 1,15,10,5;
    VectorXdual2nd x2 = y.segment(0,3);// 注意阶数匹配！！！
    //x2 << 1,10,100;
    dual2nd vd2; // 不是dual
    VectorXdual2nd gd2; // 不是VectorXdual
    MatrixXd h = hessian([&data](const VectorXdual2nd& para){return test<dual2nd>(para,data);}, wrt(x2), at(x2),vd2,gd2);
    double v2 = val(vd2);// 必须显式转换
    VectorXd g2(gd2.size());// 必须显式转换
    for(Eigen::Index i = 0; i < gd2.size(); i++){
        g2[i] = val(gd2[i]);
    }
    cout << v2 << endl << g2 << endl << h << endl;

    return 0;
}
