#include <iostream>
#include <include/pybind11/pybind11.h>
namespace py = pybind11;
using namespace std;

struct ExternData{
    int a;
    ~ExternData(){
        cout << "析构" << endl;
    }
}



int main(){
    ExternData *a = new ExternData(1);
    py::object b = py::cast(a);
}
