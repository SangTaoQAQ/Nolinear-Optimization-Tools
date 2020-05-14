//Gauss-Newton method to solve the nonlinear optimization question

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {

    double a_real = 1.0,b_real = 2.0,c_real = 1.0;//real values
    double a_e = 2.0,b_e = -1.0,c_e = 5.0;//estimation values
    double sigma = 1;


    cv::RNG rng;

    //y=exp(ax^2+bx+c)+w;
    const int data_count = 100;
    const int iteration = 200;
    //random x ,y
    vector<double> y,x;//use double rather than int
    for (int i = 0;i<data_count;i++)
    {
        double temp_x,temp_y;
        temp_x = i/100.0 ;//random x
        temp_y = exp(a_real*temp_x*temp_x+b_real*temp_x+c_real)+rng.gaussian(sigma);

        x.push_back(temp_x);
        y.push_back(temp_y);
    }
//    for(auto item: y)
//    {
//        cout<<item<<endl;
//    }

    double cost = 0,last_cost = 0;
    for (int iter = 0;iter<iteration;iter++)
    {
        Vector3d b = Vector3d::Zero();
        Matrix3d H = Matrix3d::Zero();
        cost = 0;

        Vector3d J;
        for (int i=0; i < data_count; i++)//data i must be zero at the beginning
        {
            double error = y[i] - exp(a_e * x[i] * x[i] + b_e * x[i] + c_e);
            J[0] = -x[i] * x[i] * exp(a_e * x[i] * x[i] + b_e * x[i] + c_e);
            J[1] = -x[i] * exp(a_e * x[i] * x[i] + b_e * x[i] + c_e);
            J[2] = -exp(a_e * x[i] * x[i] + b_e * x[i] + c_e);

            H += J * J.transpose();
            b += -J * error;
            cost += error * error;
        }

        cout<<"Iteration "<<iter<<endl;
        if(iter>0 && cost >= last_cost)
        {
            break;
            cout<<"Iteration finished."<<endl;
        }

        last_cost = cost;

        Vector3d delta_x;
        delta_x = H.ldlt().solve(b);


        a_e += delta_x[0];
        b_e += delta_x[1];
        c_e += delta_x[2];


        //valuate the dx is small enough
        cout << "\n a = " << a_e << "\n b = "<<b_e<<"\n c = "<<c_e<<endl;
    }


    cout << "a after estimation is " << a_e << endl;
    cout << "b after estimation is " << b_e << endl;
    cout << "c after estimation is " << c_e << endl;
    return 0;
}
