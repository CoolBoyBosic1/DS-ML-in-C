#include <iostream>
#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

struct Sample {
    vector<double> xs;
    int label;
};

//centring data
vector<Sample> centr(const vector<Sample>& S) {
    int dim = S[0].xs.size();
    vector<double> cent(dim, 0.0);

    for (const auto& s : S) {
        for (int j = 0; j < dim; j++) {
            cent[j] += s.xs[j];
        }
    }
    for (int j = 0; j < dim; j++) {
        cent[j] /= S.size();
    }
    vector<Sample> samples = S;
    for (auto& s : samples) {
        for (int j = 0; j < dim; j++) {
            s.xs[j] = s.xs[j] - cent[j];
        }
    }
    return samples;
}

//computing covariance matrix
vector<vector<double>> cov_matr(const vector<Sample>& S) {
    int dim = S[0].xs.size();
    vector<vector<double>> cov(dim, vector<double>(dim, 0.0));

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            double sum = 0.0;
            for (int k = 0; k < S.size(); k++) {
                sum += S[k].xs[i] * S[k].xs[j];
            }
            cov[i][j] = sum / (S.size() - 1);
        }
    }
    return cov;
}

MatrixXd computeCovarianceMatrix(const vector<Sample>& S) {
    vector<vector<double>> cov = cov_matr(S);
    int dim = cov.size();
    MatrixXd covMat(dim, dim);
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < dim; j++){
            covMat(i, j) = cov[i][j];
        }
    }
    return covMat;
}

//eigen values computing, sorting and main feature selection
MatrixXd getPrincipalComponents(const MatrixXd& covMat, int k) {
    SelfAdjointEigenSolver<MatrixXd> solver(covMat);
    if (solver.info() != Success) {
        cout << "Eigen decomposition failed." << endl;
        exit(1);
    }
    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();
    int n = eigenvalues.size();
    vector<pair<double, VectorXd>> eigenPairs;
    for (int i = 0; i < n; i++) {
        eigenPairs.push_back({eigenvalues(i), eigenvectors.col(i)});
    }
    sort(eigenPairs.begin(), eigenPairs.end(), [](const pair<double, VectorXd>& a, const pair<double, VectorXd>& b) {
        return a.first > b.first;
    });
    MatrixXd principalComponents(eigenvectors.rows(), k);
    for (int i = 0; i < k; i++) {
        principalComponents.col(i) = eigenPairs[i].second;
    }
    return principalComponents;
}

//data projection
vector<vector<double>> projectData(const vector<Sample>& S, const MatrixXd& principalComponents) {
    int n = S.size();
    int k = principalComponents.cols();
    int dim = principalComponents.rows();
    vector<vector<double>> projected(n, vector<double>(k, 0.0));
    for (int i = 0; i < n; i++) {
        VectorXd sampleVec(dim);
        for (int j = 0; j < dim; j++) {
            sampleVec(j) = S[i].xs[j];
        }
        VectorXd proj = principalComponents.transpose() * sampleVec;
        for (int j = 0; j < k; j++) {
            projected[i][j] = proj(j);
        }
    }
    return projected;
}

int main(){
    vector<Sample> samples = {
        { {1.0, 2.0}, 0 },
        { {3.0, 4.0}, 0 },
        { {5.0, 6.0}, 0 }
    };

    vector<Sample> centeredSamples = centr(samples);

    MatrixXd covMat = computeCovarianceMatrix(centeredSamples);

    int k = 1;
    MatrixXd principalComponents = getPrincipalComponents(covMat, k);

    vector<vector<double>> projectedData = projectData(centeredSamples, principalComponents);

    cout << "Projected data:" << endl;
    for (const auto& proj : projectedData) {
        for (auto val : proj) {
            cout << val << " ";
        }
        cout << endl;
    }
    return 0;
}
