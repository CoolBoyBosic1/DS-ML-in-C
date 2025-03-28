#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

//define loss function
typedef double (*LossFunction)(double, double);

//store gradients for parameters
struct Gradients {
    double gradSlope;
    double gradIntercept;
};

//models
//linear regression
double LinearRegression(double x, double slope, double intercept) {
    return slope * x + intercept;
}

//Loss functions
//mean squared error
double MSE(double y_pred, double y_true) {
    return pow((y_pred - y_true), 2);
}

//mean absolute error
double MAE(double y_pred, double y_true) {
    return fabs(y_pred - y_true);
}

//root mean squared error
double RMSE(double y_pred, double y_true) {
    return sqrt(pow(y_pred - y_true, 2));
}

//compute gradients using the central difference method
Gradients computeGradients(LossFunction lossFunc, const vector<double>& xs, const vector<double>& ys,
                           double slope, double intercept, double h = 1e-5) {
    Gradients grads = {0.0, 0.0};
    int n = xs.size();
    for (int i = 0; i < n; i++) {
        double x = xs[i];
        double y_true = ys[i];
        //gradient by slope
        double y_pred_plus = LinearRegression(x, slope + h, intercept);
        double y_pred_minus = LinearRegression(x, slope - h, intercept);
        double loss_plus = lossFunc(y_pred_plus, y_true);
        double loss_minus = lossFunc(y_pred_minus, y_true);
        grads.gradSlope += (loss_plus - loss_minus) / (2 * h);

        //gradient by intercept
        y_pred_plus = LinearRegression(x, slope, intercept + h);
        y_pred_minus = LinearRegression(x, slope, intercept - h);
        loss_plus = lossFunc(y_pred_plus, y_true);
        loss_minus = lossFunc(y_pred_minus, y_true);
        grads.gradIntercept += (loss_plus - loss_minus) / (2 * h);
    }
    grads.gradSlope /= n;
    grads.gradIntercept /= n;
    return grads;
}

//generalized gradient with iterations of learning
void gradientDescent(LossFunction lossFunc, const vector<double>& xs, const vector<double>& ys,
                     double &slope, double &intercept,
                     double learningRate, int iterations) {
    int n = xs.size();
    for (int i = 0; i < iterations; i++) {
        double totalLoss = 0;
        for (int j = 0; j < n; j++) {
            double y_pred = LinearRegression(xs[j], slope, intercept);
            totalLoss += lossFunc(y_pred, ys[j]);
        }
        double loss = totalLoss / n;
        Gradients grads = computeGradients(lossFunc, xs, ys, slope, intercept);

        //updating parameters
        slope -= learningRate * grads.gradSlope;
        intercept -= learningRate * grads.gradIntercept;

        if (i % 100 == 0) {
            cout << "Iteration " << i
                 << " | Loss: " << loss
                 << " | Slope: " << slope
                 << " | Intercept: " << intercept << endl;
        }
    }
}

int main() {
    //data sample
    vector<double> xs = {2.0, 3.0, 4.0, 5.0};
    vector<double> ys = {3.0, 4.0, 5.0, 6.0};

    //model parameters
    double slope = 1.5, intercept = 0.5;

    //hyperparameters for gradient descent
    double learningRate = 0.01;
    int iterations = 1000;

    //loss function - MSE
    cout << "Using MSE :" << endl;
    gradientDescent(MSE, xs, ys, slope, intercept, learningRate, iterations);
    cout << "Final model (MSE): y = " << slope << " * x + " << intercept << "\n" << endl;

    //reset params
    slope = 1.5;
    intercept = 0.5;

    //loss function - MAE
    cout << "Using MAE:" << endl;
    gradientDescent(MAE, xs, ys, slope, intercept, learningRate, iterations);
    cout << "Final model (MAE): y = " << slope << " * x + " << intercept << "\n" << endl;

    //reset params
    slope = 1.5;
    intercept = 0.5;

    //loss function - RMSE
    cout << "Using RMSE:" << endl;
    gradientDescent(RMSE, xs, ys, slope, intercept, learningRate, iterations);
    cout << "Final model (RMSE): y = " << slope << " * x + " << intercept << endl;

    return 0;
}
