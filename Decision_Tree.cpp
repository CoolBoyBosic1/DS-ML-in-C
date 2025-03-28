#include <iostream>
#include <bits/stdc++.h>
using namespace std;

struct Sample {
    vector<double> features;
    int label;
};

//frequency per value
map<double, double> var_in_s(const vector<double>& x) {
    map<double, int> freq;
    for (int i = 0; i < x.size(); i++){
        freq[x[i]]++;
    }
    //normalize freq
    map<double, double> normFreq;
    for (auto &p : freq) {
        normFreq[p.first] = static_cast<double>(p.second) / x.size();
    }
    return normFreq;
}

//loss function: entropy
double entropy(const vector<Sample>& S) {
    vector<double> labels;
    for (int i = 0; i < S.size(); i++){
        labels.push_back(S[i].label);
    }
    map<double, double> freq_entrop = var_in_s(labels);
    double sum = 0.0;
    for (auto &p : freq_entrop) {
        double p_val = p.second;
        if (p_val > 0)
            sum += p_val * (log(p_val) / log(2.0));
    }
    return -sum;
}

//loss function: djini
double djini(const vector<Sample>& S) {
    vector<double> labels;
    for (int i = 0; i < S.size(); i++){
        labels.push_back(S[i].label);
    }
    map<double, double> freq_djini = var_in_s(labels);
    double sum = 0.0;
    for (auto &p : freq_djini) {
        sum += p.second * p.second;
    }
    return 1 - sum;
}

// splitting tree function
//for each feature, try candidate thresholds (midpoints) and compute information gain
void split(const vector<Sample>& S) {
    if(S.empty()) return;
    int nFeatures = S[0].features.size();
    double baseEntropy = entropy(S);

    double bestInfoGain = -numeric_limits<double>::infinity();
    int bestFeature = -1;
    double bestThreshold = 0.0;
    vector<Sample> bestLeft, bestRight;

    for (int f = 0; f < nFeatures; f++) {
        vector<double> candidates;
        for (int i = 0; i < S.size(); i++){
            candidates.push_back(S[i].features[f]);
        }
        sort(candidates.begin(), candidates.end());
        candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());

        for (int i = 0; i < candidates.size() - 1; i++){
            double threshold = (candidates[i] + candidates[i+1]) / 2.0;
            vector<Sample> left, right;
            for (int j = 0; j < S.size(); j++){
                if (S[j].features[f] <= threshold)
                    left.push_back(S[j]);
                else
                    right.push_back(S[j]);
            }
            if(left.empty() || right.empty())
                continue;

            double leftEntropy = entropy(left);
            double rightEntropy = entropy(right);
            double weightedEntropy = (left.size() / (double)S.size()) * leftEntropy + (right.size() / (double)S.size()) * rightEntropy;
            double infoGain = baseEntropy - weightedEntropy;

            if(infoGain > bestInfoGain) {
                bestInfoGain = infoGain;
                bestFeature = f;
                bestThreshold = threshold;
                bestLeft = left;
                bestRight = right;
            }
        }
    }

    cout << "Best split:" << endl;
    cout << "Feature index: " << bestFeature << endl;
    cout << "Threshold: " << bestThreshold << endl;
    cout << "Information Gain: " << bestInfoGain << endl;
    cout << "Left samples: " << bestLeft.size() << ", Right samples: " << bestRight.size() << endl;
}

int main() {
    vector<Sample> samples = {
        { {2.0, 3.0}, 0 },
        { {2.5, 3.5}, 0 },
        { {3.0, 4.0}, 1 },
        { {3.5, 4.5}, 1 },
        { {4.0, 5.0}, 1 }
    };

    split(samples);

    return 0;
}
