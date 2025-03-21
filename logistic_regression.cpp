#include <iostream>
#include <bits/stdc++.h>
using namespace std;

//model - logistic regression
double LogisticRegression(double x, double slope, double intercept){
    return 1 / (1 + exp(-(x * slope + intercept)));
}

//loss function - crossentropy
double Crossentropy(double y_test, double p){
    return -(y_test * log(p) + (1 - y_test) * log(1 - p));
}

//optimization function - RMSprop
class RMSprop{
public:
    void refresh_exp_med(double gradient, double decay, double &exp_med) {
        exp_med = decay * exp_med + (1 - decay) * (gradient * gradient);
    }

    double param_update(double param, double grad, double learning_rate, double &exp_med){
        double e = 0.00000001;
        return param - (learning_rate * grad) / sqrt(exp_med + e);
    }
};

int main()
{
    vector<double> xs = {2.0, 3.0, 4.0, 5.0};
    vector<double> ys = {0, 0, 1, 1};

    double slope = 0.0, intercept = 0.0;
    double learning_rate = 0.01;
    int iterations = 1000;
    double decay = 0.9;

    double exp_med_slope = 0.0;
    double exp_med_intercept = 0.0;

    RMSprop optimizer;

    for (int iter = 0; iter < iterations; iter++){
        double total_loss = 0.0;
        double grad_slope_sum = 0.0;
        double grad_intercept_sum = 0.0;
        int n = xs.size();

        for (int i = 0; i < n; i++){
            double p = LogisticRegression(xs[i], slope, intercept);
            total_loss += Crossentropy(ys[i], p);
            double grad = p - ys[i];
            grad_slope_sum += grad * xs[i];
            grad_intercept_sum += grad;
        }

        double avg_loss = total_loss / n;
        double grad_slope = grad_slope_sum / n;
        double grad_intercept = grad_intercept_sum / n;

        optimizer.refresh_exp_med(grad_slope, decay, exp_med_slope);
        optimizer.refresh_exp_med(grad_intercept, decay, exp_med_intercept);

        slope = optimizer.param_update(slope, grad_slope, learning_rate, exp_med_slope);
        intercept = optimizer.param_update(intercept, grad_intercept, learning_rate, exp_med_intercept);

        if (iter % 100 == 0) {
            cout << "Iteration " << iter
                 << " | Loss: " << avg_loss
                 << " | Slope: " << slope
                 << " | Intercept: " << intercept << endl;
        }
    }

    cout << "Final model parameters:" << endl;
    cout << "Slope: " << slope << ", Intercept: " << intercept << endl;

    return 0;
}
