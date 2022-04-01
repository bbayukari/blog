#include<Eigen/Dense>
#include<iostream>
#include<vector>
using namespace std;
using namespace Eigen;

int main(){
    double n[5]{1.1,2.2,3.3,4.4,5.5};
    double const *x = n;
    Map<VectorXd const> v(x,5);
    MatrixXd m = MatrixXd::Ones(5,5);
    cout << m*v; 
    return 0;
}
