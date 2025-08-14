#include <cmath>
#include <iostream>
#include <vector>
#include "genetic.h"

using namespace std;

constexpr double PI = 3.14159265358979323846;

double fit(const vector<double> &params) {
    double x = params[0], y = params[1];
    double xp = x - PI;
    double yp = y - PI;
    return -cos(x) * cos(y) * exp(-(xp * xp + yp * yp));
}

// pretty print
void pp(vector<double> &p) {
    for (auto i : p)
    cout << i << " ";
    cout << endl;
}

int main(int argc, char *argv[]) {
    if (argc != 3)
        return 0;
    int G = stoi(argv[1]);
    int P = stoi(argv[2]);

    optimize::Engine<> engine(P, G,
                              optimize::randomize::randvector(2),
                              fit,
                              optimize::crossbreed::BLX_a(),
                              optimize::mutate::simple(),
                              optimize::select::tournament(P / 5));

    auto p = engine.simulate();

    double Mx = fit(p[0]);
    cout << Mx << " - ";
    pp(p[0]);

    return 0;
}
