#include <iostream>
#include <bits/stdc++.h>
#include <cmath>
using namespace std;

struct Sample{
    vector<vector<double>> xs;
    vector<double> ys;
};

//activation functions:
//Rectified linear unit
double RELU(double x){
    return max(0.0, x);
}
//sigmoid
double SG(double x){
    return 1 / (1 + exp(-x));
}
//tanhensoid
double TANH(double x){
    return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

//finite difference
double num_derivative(double (*act_func)(double), double x){
    double h = 1e-6;
    return (act_func(x + h) - act_func(x - h)) / (2 * h);
}

//forward pass for two layer MLP (1 neuron in each layer
double forwardNetwork(double input, double w1, double b1, double w2, double b2){
    double z1 = w1 * input + b1;
    double a1 = RELU(z1); //choose activation function(RELU, SG, TANH)
    double z2 = w2 * a1 + b2;
    double a2 = RELU(z2); //choose activation function(RELU, SG, TANH)
    return a2;
}

//backpropagation
void trainNetwork(double input, double target,
                  double &w1, double &b1,
                  double &w2, double &b2, double lr) {
    // forward pass
    double z1 = w1 * input + b1;
    double a1 = RELU(z1);
    double z2 = w2 * a1 + b2;
    double a2 = RELU(z2);

    double error = target - a2;

    double d_act2 = num_derivative(RELU, z2);
    double delta2 = error * d_act2;

    double d_act1 = num_derivative(RELU, z1);
    double delta1 = w2 * delta2 * d_act1;

    //param update
    w2 += lr * delta2 * a1;
    b2 += lr * delta2;
    w1 += lr * delta1 * input;
    b1 += lr * delta1;
}

int main(){
    //train data
    double input = 1.0;
    double target = 0.5;

    //initialize parameters
    double w1 = 0.1, b1 = 0.1;
    double w2 = 0.2, b2 = 0.2;
    double lr = 0.01;

    //training cycle
    for(int epoch = 0; epoch < 1000; epoch++){
        trainNetwork(input, target, w1, b1, w2, b2, lr);
        if(epoch % 100 == 0){
            double out = forwardNetwork(input, w1, b1, w2, b2);
            cout << "Epoch " << epoch << " Loss: " << pow(target - out, 2) << endl;
        }
    }
    cout << "Final output: " << forwardNetwork(input, w1, b1, w2, b2) << endl;
    return 0;
}
