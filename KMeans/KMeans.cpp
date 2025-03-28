#include <iostream>
#include <bits/stdc++.h>
using namespace std;

struct Sample {
    vector<double> features;
    int label;
};

// k-means clustering algorithm
vector<vector<double>> kmeans(const vector<Sample>& S, int n_centroids, int iterations) {
    int n = S.size();
    int dim = S[0].features.size();

    vector<vector<double>> centroids(n_centroids, vector<double>(dim, 0.0));
    for (int i = 0; i < n_centroids; i++){
        int idx = rand() % n;
        centroids[i] = S[idx].features;
    }

    vector<int> assignments(n, -1);

    for (int iter = 0; iter < iterations; iter++){
        for (int i = 0; i < n; i++){
            double minDist = numeric_limits<double>::max();
            int bestCluster = -1;
            for (int j = 0; j < n_centroids; j++){
                double dist = 0.0;
                for (int d = 0; d < dim; d++){
                    dist += pow(S[i].features[d] - centroids[j][d], 2);
                }
                if (dist < minDist){
                    minDist = dist;
                    bestCluster = j;
                }
            }
            assignments[i] = bestCluster;
        }

        vector<vector<double>> newCentroids(n_centroids, vector<double>(dim, 0.0));
        vector<int> counts(n_centroids, 0);
        for (int i = 0; i < n; i++){
            int cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < dim; d++){
                newCentroids[cluster][d] += S[i].features[d];
            }
        }
        for (int j = 0; j < n_centroids; j++){
            if (counts[j] > 0) {
                for (int d = 0; d < dim; d++){
                    newCentroids[j][d] /= counts[j];
                }
            } else {
                int idx = rand() % n;
                newCentroids[j] = S[idx].features;
            }
        }
        centroids = newCentroids;
    }

    return centroids;
}

int main() {
    vector<Sample> samples = {
        { {1.0, 2.0}, 0 },
        { {1.2, 1.8}, 0 },
        { {5.0, 8.0}, 1 },
        { {6.0, 9.0}, 1 },
        { {1.5, 2.2}, 0 },
        { {5.5, 7.8}, 1 }
    };

    int k = 2;
    int iterations = 100;

    vector<vector<double>> centroids = kmeans(samples, k, iterations);

    for (int i = 0; i < centroids.size(); i++){
        cout << "Centroid " << i << ": ";
        for (double val : centroids[i])
            cout << val << " ";
        cout << endl;
    }

    return 0;
}
