#include <iostream>
#include <bits/stdc++.h>
using namespace std;

struct Sample{
    vector<double> features;
    int label;
};

//if point is in our reach of eps
bool in_eucl_dist(const vector<double>& xs, const vector<double>& ys, double eps) {
    double sum = 0.0;
    for(int i = 0; i < xs.size(); i++){
        sum += pow(xs[i] - ys[i], 2);
    }
    return sqrt(sum) < eps;
}

//neighbours within eps
vector<int> get_neighbours(const vector<Sample>& S, int ind, double eps) {
    vector<int> neighbours;
    for(int i = 0; i < S.size(); i++){
        if(in_eucl_dist(S[ind].features, S[i].features, eps)) {
            neighbours.push_back(i);
        }
    }
    return neighbours;
}

//expanding cluster
void expandCluster(const vector<Sample>& S, int ind, int clustid, double eps, int minpts, vector<int>& clustassign, vector<bool>& visited) {
    vector<int> neighbours = get_neighbours(S, ind, eps);

    if(neighbours.size() < minpts) {
        clustassign[ind] = -1;
        return;
    }

    clustassign[ind] = clustid;

    for (int i = 0; i < neighbours.size(); i++) {
        int nb = neighbours[i];
        if (!visited[nb]) {
            visited[nb] = true;
            vector<int> nbneighbours = get_neighbours(S, nb, eps);
            if (nbneighbours.size() >= minpts) {
                neighbours.insert(neighbours.end(), nbneighbours.begin(), nbneighbours.end());
            }
        }
        if (clustassign[nb] == 0) {
            clustassign[nb] = clustid;
        }
    }
}

vector<int> DBSCAN(const vector<Sample>& S, double eps, int minpts) {
    int n = S.size();
    vector<int> clustassign(n, 0);
    vector<bool> visited(n, false);
    int clustid = 0;

    for (int i = 0; i < n; i++){
        if (!visited[i]) {
            visited[i] = true;
            vector<int> neighbours = get_neighbours(S, i, eps);
            if (neighbours.size() < minpts) {
                clustassign[i] = -1;
            } else {
                clustid++;
                expandCluster(S, i, clustid, eps, minpts, clustassign, visited);
            }
        }
    }

    return clustassign;
}

int main()
{
    vector<Sample> samples = {
        { {1.0, 2.0}, 0 },
        { {1.1, 2.1}, 0 },
        { {5.0, 8.0}, 0 },
        { {5.1, 7.9}, 0 },
        { {10.0, 12.0}, 0 }
    };

    double eps = 1.5;
    int minpts = 2;

    vector<int> clusters = DBSCAN(samples, eps, minpts);

    for (int i = 0; i < clusters.size(); i++){
        cout << "Sample " << i << " assigned to cluster: " << clusters[i] << endl;
    }

    return 0;
}
