#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "genetic.h"

using namespace std;

constexpr double PI = 3.14159265358979323846;

struct dwrap {
    double x;

    dwrap(double a) : x(a) {}

    dwrap() : x(0) {}

    bool operator<(const dwrap& a) const {
        return x < a.x;
    }

    dwrap operator*(const dwrap& a) const {
        return { x * a.x };
    }

    dwrap operator+(const dwrap& a) const {
        return { x + a.x };
    }
};


double fit(const vector<dwrap> &params) {
    double x = params[0].x, y = params[1].x;
    double xp = x - PI;
    double yp = y - PI;
    return -cos(x) * cos(y) * exp(-(xp * xp + yp * yp));
}

vector<dwrap> cross(const vector<dwrap> &a, const vector<dwrap> &b, [[maybe_unused]] pcg::pcg gen, [[maybe_unused]] const optimize::Engine<vector<dwrap>>& eng) {
    vector<dwrap> child {a[0], b[1]};
    return child;
}

void mut(vector<dwrap>& a, pcg::pcg gen, [[maybe_unused]] const optimize::Engine<vector<dwrap>>& eng) {
    std::normal_distribution<> gauss(0, 0.5);
    for(auto& i : a) i.x += gauss(gen);
}

vector<dwrap> my_rand(pcg::pcg gen, [[maybe_unused]] const optimize::Engine<vector<dwrap>>& eng) {
    std::uniform_real_distribution<> init_uni{-1.0, 1.0};
    const auto N = 2;
    std::vector<dwrap> x(N);
    for (size_t i = 0; i < N; ++i) x[i].x = init_uni(gen);
    return x;
}

// pretty print
void pp(vector<dwrap> &p) {
    for (auto i : p)
    cout << i.x << " ";
    cout << endl;
}

int main(int argc, char *argv[]) {
    if (argc != 3)
        return 0;
    int G = stoi(argv[1]);
    int P = stoi(argv[2]);

    using eng = optimize::Engine<vector<dwrap>>;
    eng engine(P, G, my_rand, fit, cross, mut, optimize::select::tournament<double, eng>(P / 5));

    auto p = engine.simulate();

    dwrap Mx = fit(p[0]);
    cout << Mx.x << " - ";
    pp(p[0]);

    return 0;
}
