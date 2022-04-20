#include<Eigen/Dense>
#include<iostream>
#include<vector>
using namespace std;
using namespace Eigen;

void foo(Map<VectorXd>& a){
    VectorXd c(5);
    c << 10,20,30,40,50;
    a = c;
}

int main(){
    VectorXd x(3), y(3);
    MatrixXd m(3,3);
    x << 1,2,1;
    y << 10,20,30;
    m << 1,0,0,0,1,0,0,0,1;
    y(0) = (x.transpose() * m * x)(0) /1;
    cout << y(x);

    cout << y;
    return 0;
}
