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

class RandomForest {
public:
    //Bootstrap from our dataset using rand (sampling with replacement)
    vector<vector<Sample>> Bootstrap(const vector<Sample>& s, int B) {
        vector<vector<Sample>> subsamples;
        subsamples.resize(B);
        int n = s.size();

        for (int i = 0; i < B; i++){
            for (int j = 0; j < n; j++){
                int idx = rand() % n;
                subsamples[i].push_back(s[idx]);
            }
        }
        return subsamples;
    }

    //voting between decision trees
    int MajorityVote(const vector<Sample>& s, int numTrees) {
        DecisionTree dt;
        vector<int> votes;
        vector<vector<Sample>> subsamples = Bootstrap(s, numTrees);
        for (int i = 0; i < numTrees; i++) {
            SplitResult res = dt.split(subsamples[i]);
            map<int, int> freq;
            for (auto sample : res.left) {
                freq[sample.label]++;
            }
            int bestLabel = -1, maxCount = 0;
            for (auto &p : freq) {
                if (p.second > maxCount) {
                    maxCount = p.second;
                    bestLabel = p.first;
                }
            }
            votes.push_back(bestLabel);
        }
        map<int, int> voteCount;
        for (int v : votes) {
            voteCount[v]++;
        }
        int finalLabel = -1, maxVotes = 0;
        for (auto &p : voteCount) {
            if (p.second > maxVotes) {
                maxVotes = p.second;
                finalLabel = p.first;
            }
        }
        return finalLabel;
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

    DecisionTree dt;
    SplitResult bestSplit = dt.split(samples);
    cout << "Best split:" << endl;
    cout << "Feature index: " << bestSplit.bestFeature << endl;
    cout << "Threshold: " << bestSplit.bestThreshold << endl;
    cout << "Information Gain: " << bestSplit.infoGain << endl;
    cout << "Left samples: " << bestSplit.left.size() << ", Right samples: " << bestSplit.right.size() << endl;

    RandomForest rf;
    int predicted = rf.MajorityVote(samples, 3);
    cout << "Predicted label (via Majority Vote): " << predicted << endl;

    return 0;
}
