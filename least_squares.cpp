#include <iostream>
#include <bits/stdc++.h>
using namespace std;

class LeastSquares{
public:
    double average(const vector<int>& X){
        double sum = 0;
        for (int i = 0; i < X.size(); i++){
            sum += X[i];
        }
        return sum / X.size();
    }

    double variance(const vector<int>& x){
        double sum = 0;
        double avg = average(x);
        for (int i = 0; i < x.size(); i++){
            sum += pow(x[i] - avg, 2);
        }
        return sum;
    }

    double composition_dispertion(const vector<int>& x, const vector<int>& y){
        double sum = 0;
        double avg_x = average(x);
        double avg_y = average(y);
        for (int i = 0; i < x.size(); i++){
            sum += (x[i] - avg_x) * (y[i] - avg_y);
        }
        return sum;
    }

    double param_update_b1(const vector<int>& x, const vector<int>& y){
        return composition_dispertion(x, y) / variance(x);
    }

    double param_update_b0(const vector<int>& x, const vector<int>& y){
        return average(y) - average(x) * param_update_b1(x, y);
    }

    double linear_model_point(int x, const vector<int>& X, const vector<int>& Y){
        return param_update_b0(X, Y) + param_update_b1(X, Y) * x;
    }
};

int main()
{
    vector<int> xs = {1, 2, 3, 4, 5};
    vector<int> ys = {2, 4, 5, 4, 5};

    LeastSquares ls;

    double b1 = ls.param_update_b1(xs, ys);
    double b0 = ls.param_update_b0(xs, ys);

    cout << "Calculated parameters:" << endl;
    cout << "b0 (intercept): " << b0 << endl;
    cout << "b1 (slope): " << b1 << endl;

    cout << "\nPredictions:" << endl;
    for (int i = 0; i < xs.size(); i++){
        cout << "x = " << xs[i] << " -> y_pred = " << ls.linear_model_point(xs[i], xs, ys) << endl;
    }

    return 0;
}
