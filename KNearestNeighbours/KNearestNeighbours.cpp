#include <iostream>
#include <bits/stdc++.h>
using namespace std;

struct point {
    vector<double> features;
    int label;
};

//distance between points
double euclidian_distance(vector<double> xs, vector<double> ys) {
    double sum = 0.0;
    for (int i = 0; i < xs.size(); i++) {
        sum += pow(xs[i] - ys[i], 2);
    }
    return sqrt(sum);
}

//K-nearest neighbours
int KNN(const vector<point>& training, const point& test, int k) {
    vector<pair<double, int>> distances;
    for (int i = 0; i < training.size(); i++) {
        double d = euclidian_distance(training[i].features, test.features);
        distances.push_back(make_pair(d, training[i].label));
    }
    sort(distances.begin(), distances.end(), [](pair<double, int> a, pair<double, int> b) {
        return a.first < b.first;
    });
    map<int, int> freq;
    for (int i = 0; i < k; i++) {
        freq[distances[i].second]++;
    }
    int max_count = 0, predicted = -1;
    for (auto &p : freq) {
        if (p.second > max_count) {
            max_count = p.second;
            predicted = p.first;
        }
    }
    return predicted;
}

int main() {
    vector<point> training;
    point p;

    p.features = {1.0, 2.0}; p.label = 0; training.push_back(p);
    p.features = {2.0, 3.0}; p.label = 0; training.push_back(p);
    p.features = {3.0, 3.0}; p.label = 1; training.push_back(p);
    p.features = {6.0, 7.0}; p.label = 1; training.push_back(p);

    point test;
    test.features = {2.5, 3.0};

    int k = 3;
    int label = KNN(training, test, k);
    cout << "Predicted label: " << label << endl;

    return 0;
}
