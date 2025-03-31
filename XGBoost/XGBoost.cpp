#include <iostream>
#include <bits/stdc++.h>
using namespace std;

struct Sample {
    vector<double> features;
    int label;
};

struct SplitResult {
    int bestFeature;
    double bestThreshold;
    double infoGain;
    vector<Sample> left;
    vector<Sample> right;
};

class DecisionTree {
public:
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
    SplitResult split(const vector<Sample>& S) {
        SplitResult best;
        if(S.empty()) return best;
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
                double weightedEntropy = (left.size() / (double)S.size()) * leftEntropy +
                                           (right.size() / (double)S.size()) * rightEntropy;
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

        best.bestFeature = bestFeature;
        best.bestThreshold = bestThreshold;
        best.infoGain = bestInfoGain;
        best.left = bestLeft;
        best.right = bestRight;
        return best;
    }
};

class XGBoost {
public:
    vector<SplitResult> trees;
    vector<double> predictions;
    vector<Sample> trainingData;
    double learning_rate;
    int numTrees;

    //initialize training data, learning rate, and number of boosting rounds
    XGBoost(const vector<Sample>& s, double lr, int nT) {
        trainingData = s;
        learning_rate = lr;
        numTrees = nT;
        predictions.resize(s.size(), 0.0);
    }

    //objective function: compute the current RMSE of predictions
    double Objective_func() {
        double sum = 0.0;
        int n = trainingData.size();
        for (int i = 0; i < n; i++) {
            double err = trainingData[i].label - predictions[i];
            sum += err * err;
        }
        return sqrt(sum / n);
    }

    //decision tree on the residuals
    SplitResult BuildTree() {
        vector<Sample> residualSamples = trainingData;
        for (int i = 0; i < residualSamples.size(); i++) {
            double error = trainingData[i].label - predictions[i];
            residualSamples[i].label = (error >= 0) ? 1 : 0;
        }
        DecisionTree dt;
        return dt.split(residualSamples);
    }

    //update predictions with the output of a new tree
    void UpdatePredictions(const SplitResult& tree) {
        double offset = 0.0;
        if (!tree.left.empty()) {
            for (int i = 0; i < tree.left.size(); i++) {
                offset += tree.left[i].label;
            }
            offset /= tree.left.size();
        }

        for (int i = 0; i < predictions.size(); i++) {
            predictions[i] += learning_rate * offset;
        }
    }

    void BoostingLoop() {
        for (int t = 0; t < numTrees; t++) {
            SplitResult tree = BuildTree();
            trees.push_back(tree);
            UpdatePredictions(tree);
            cout << "Tree " << t << " built, Objective: " << Objective_func() << endl;
        }
    }
};

int main() {
    vector<Sample> samples = {
        { {2.0, 3.0}, 0 },
        { {2.5, 3.5}, 0 },
        { {3.0, 4.0}, 1 },
        { {3.5, 4.5}, 1 },
        { {4.0, 5.0}, 1 }
    };

    XGBoost xgb(samples, 0.1, 5);
    xgb.BoostingLoop();

    cout << "Final Objective: " << xgb.Objective_func() << endl;

    return 0;
}
