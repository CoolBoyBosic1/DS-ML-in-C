#include <iostream>
#include <bits/stdc++.h>
using namespace std;

struct Sample {
    vector<double> features;
    int label;
};

//aprior probabilities
map<int, double> aprior_prob(const vector<Sample>& samples) {
    map<int, int> count;
    for (const auto &s : samples)
        count[s.label]++;
    map<int, double> prob;
    for (auto &p : count)
        prob[p.first] = static_cast<double>(p.second) / samples.size();
    return prob;
}

//discrete conditional probability
map<int, map<double, double>> disc_cond_prob(const vector<Sample>& samples, int featureIndex) {
    map<int, map<double, int>> count;
    map<int, int> total;
    for (const auto &s : samples) {
        count[s.label][s.features[featureIndex]]++;
        total[s.label]++;
    }
    map<int, map<double, double>> condProb;
    for (auto &entry : count) {
        int label = entry.first;
        for (auto &val : entry.second)
            condProb[label][val.first] = static_cast<double>(val.second) / total[label];
    }
    return condProb;
}

// continuous conditional probability using KDE
map<int, double> cont_cond_prob(const vector<Sample>& samples, int featureIndex, double x_value, double h) {
    map<int, vector<double>> values;
    for (const auto &s : samples)
        values[s.label].push_back(s.features[featureIndex]);

    auto kernel = [](double u) -> double {
        return exp(-0.5 * u * u) / sqrt(2 * M_PI);
    };

    map<int, double> condProb;
    for (auto &entry : values) {
        int label = entry.first;
        const vector<double>& vals = entry.second;
        double sum = 0.0;
        for (double v : vals) {
            double u = (x_value - v) / h;
            sum += kernel(u);
        }
        condProb[label] = sum / (vals.size() * h);
    }
    return condProb;
}


int main() {
    vector<Sample> samples = {
        { {1.0, 2.0}, 0 },
        { {1.0, 3.0}, 0 },
        { {2.0, 2.5}, 1 },
        { {2.0, 3.5}, 1 }
    };

    auto apriori = aprior_prob(samples);
    for (auto &p : apriori)
        cout << "Class " << p.first << " prior: " << p.second << endl;

    //discrete conditional probability for feature 0
    int discFeatureIndex = 0;
    auto discCond = disc_cond_prob(samples, discFeatureIndex);
    for (auto &entry : discCond) {
        cout << "Class " << entry.first << " conditional probs for feature " << discFeatureIndex << ":\n";
        for (auto &val : entry.second)
            cout << "  Value " << val.first << " -> " << val.second << endl;
    }

    //continuous conditional probability for feature 1 at test value 3.0
    int contFeatureIndex = 1;
    double testValue = 3.0;
    auto contCond = cont_cond_prob(samples, contFeatureIndex, testValue);
    for (auto &entry : contCond)
        cout << "Class " << entry.first << " density for feature " << contFeatureIndex << " at " << testValue << ": " << entry.second << endl;

    return 0;
}
