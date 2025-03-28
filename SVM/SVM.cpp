#include <iostream>
#include <bits/stdc++.h>
using namespace std;

struct Sample {
    vector<double> features;
    int label;
};

// model: support vector machine
double SVM(double w, double x, double b) {
    return w * x + b;
}

// loss: hinge loss
double HingeLoss(double w, double b, const vector<double>& xs, const vector<double>& ys) {
    double sum = 0.0;
    int n = xs.size();
    for (int i = 0; i < n; i++) {
        sum += max(0.0, 1 - ys[i] * SVM(w, xs[i], b));
    }
    return sum;
}

// regularization: L2(hingeloss)
double TotalLoss(double w, double b, const vector<double>& xs, const vector<double>& ys, double lambda) {
    return HingeLoss(w, b, xs, ys) + lambda * (w * w);
}

//optimization: gradient descent
vector<double> gradientDescent(vector<double> theta, double learning_rate, int iterations,
    function<vector<double>(const vector<double>&)> gradFunc) {
    for (int i = 0; i < iterations; i++) {
        vector<double> grad = gradFunc(theta);
        for (int j = 0; j < theta.size(); j++) {
            theta[j] -= learning_rate * grad[j];
        }
    }
    return theta;
}

int main() {
    vector<double> xs = {1.0, 2.0, 3.0, 4.0, 5.0};
    vector<double> ys = {-1, -1, 1, 1, 1};
    double lambda = 0.01;

    vector<double> theta = {0.0, 0.0};

    double learning_rate = 0.01;
    int iterations = 1000;

    //gradient for hinge loss +l2
    auto gradFunc = [&, xs, ys, lambda](const vector<double>& theta) -> vector<double> {
        double w = theta[0];
        double b = theta[1];
        double grad_w = 0.0;
        double grad_b = 0.0;
        int n = xs.size();
        //sub gradient hinge loss
        for (int i = 0; i < n; i++) {
            double y = ys[i];
            double x = xs[i];
            double prediction = SVM(w, x, b);
            if (y * prediction < 1) {
                grad_w += -y * x;
                grad_b += -y;
            }
        }
        grad_w /= n;
        grad_b /= n;
        grad_w += 2 * lambda * w;
        return vector<double>{grad_w, grad_b};
    };

    vector<double> optimalTheta = gradientDescent(theta, learning_rate, iterations, gradFunc);

    cout << "Optimal parameters:" << endl;
    cout << "w: " << optimalTheta[0] << ", b: " << optimalTheta[1] << endl;
    cout << "Final loss: " << TotalLoss(optimalTheta[0], optimalTheta[1], xs, ys, lambda) << endl;

    return 0;
}
