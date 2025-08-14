#include <cmath>
#include <iostream>
#include <vector>
#include "genetic.h"

using namespace std;

double fit(const vector<double> &params) {
    double x = params[0], y = params[1];
    return pow(x*x + y - 11, 2) + pow(x + y*y - 7, 2);
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

    optimize::Engine<> engine(P, G, optimize::randomize::randvector(2), fit, optimize::crossbreed::BLX_a(), optimize::mutate::simple(), optimize::select::tournament());

    auto p = engine.simulate();

    double Mx = fit(p[0]);
    cout << Mx << " - ";
    pp(p[0]);

    return 0;
}
