#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "genetic.h"

using namespace std;

using param_t = double;

const vector<vector<param_t>> input{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
const vector<param_t> output{0, 1, 1, 0};
const int M = 4; // number of training examples
const int N = 13; // number of parameters

inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

vector<double> eval(const vector<param_t> &params) {
    vector<double> val(M);
    for (int i = 0; i < M; i++) {
        double l11 = sigmoid(input[i][0] * params[0] + input[i][1] * params[1] + params[2]);
        double l12 = sigmoid(input[i][0] * params[3] + input[i][1] * params[4] + params[5]);
        double l13 = sigmoid(input[i][0] * params[6] + input[i][1] * params[7] + params[8]);
        double l2  = l11 * params[9] + l12 * params[10] + l13 * params[11] + params[12];
        // double l2  = l11 * params[6] + l12 * params[7] + params[8];
        val[i] = sigmoid(l2);
    }
    return val;
}

double fit(const vector<param_t> &params) {
    double f = 0.0;
    vector<double> evals = eval(params);

    // Cross-entropy loss
    for (int i = 0; i < M; i++) {
        double y = min(max(evals[i], 1e-6), 1 - 1e-6); // avoid log(0)
        f -= output[i] * log(y) + (1 - output[i]) * log(1 - y);
    }

    return f;
}

void pp(const vector<param_t> &p) {
    for (auto v : p) cout << v << " ";
    cout << endl;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <generations> <population> [seed]" << endl;
        return 1;
    }

    int G = stoi(argv[1]);
    int P = stoi(argv[2]);
    unsigned seed = (argc >= 4) ? std::stoul(argv[3]) : std::random_device{}();

    // Track the overall best across multiple runs
    double best_score = 1e9;
    vector<param_t> best_params;

    const int RUNS = 1; // multiple attempts for reliability
    for (int run = 0; run < RUNS; ++run) {
        optimize::Engine<vector<param_t>> engine(P, G,
                                         optimize::randomize::randvector<uniform_real_distribution<>, param_t>(N),
                                         fit,
                                         optimize::crossbreed::BLX_a<param_t>(),
                                         optimize::mutate::simple<vector<param_t>>(),
                                         optimize::select::tournament<double, optimize::Engine<vector<param_t>>>());
        engine.setSeed(seed + run); // different seed per run but reproducible overall

        engine.init();
        for(int _g = 0; _g < G; _g++) {
            engine.step();
        }
        auto pop = engine.get_population();
        auto scores = engine.get_scores();

        // Check the best in this population
        for (int i = 0; i < P; i++) {
            if (scores[i] < best_score) {
                best_score = scores[i];
                best_params = pop[i];
            }
        }
    }

    cout << "Best fitness: " << best_score << "\n";
    cout << "Best parameters:\n";
    pp(best_params);

    cout << "Network outputs:\n";
    auto vals = eval(best_params);
    for (auto ans : vals) cout << ans << "\n";

    return 0;
}
