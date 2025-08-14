#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>
#include <execution>

#include "pcg.h"

namespace optimize {

// ------------------------------------------
// ------ Engine -------

template <typename param_t = std::vector<double>, typename fitret_t = double>
class Engine {
public:
    using pop_t        = std::vector<param_t>; // population is an array of parameters
    using fit_t        = std::function<fitret_t(const param_t&)>; // fitness
    using score_t      = std::vector<fitret_t>; // scores is an array of return type of fitness
    using rand_t       = std::function<param_t(pcg::pcg, const Engine<param_t, fitret_t>&)>; // individual randomizer
    using crossbreed_t = std::function<param_t(const param_t&, const param_t&, pcg::pcg, const Engine<param_t, fitret_t>&)>; // crossbreeder
    using mutate_t     = std::function<void(param_t&, pcg::pcg, const Engine<param_t, fitret_t>&)>; // mutater
    using select_t     = std::function<std::vector<size_t>(size_t n, pcg::pcg, const Engine<param_t, fitret_t>&)>; // selector
    using dist_t       = std::function<bool(const param_t&, const param_t&)>; // distance between individuals

private:
    // --- Functions ---
    fit_t   fit{};
    crossbreed_t cross{};
    mutate_t mutate{};
    select_t select{};
    rand_t randomize{};
    dist_t dist{};

    // --- Knobs ---
    uint32_t P{0}, G{0};
    uint32_t K{1};         // elites per generation
    uint32_t R_base{1};    // base number of random immigrants per generation
    bool decay_R_base {true};
    uint32_t reset_freq{0};
    uint32_t log_freq{100};

    // --- Current State ---
    uint32_t cur_G;
    pop_t pop;
    score_t scores;
    std::vector<size_t> idx;

    // --- RNG ---
    std::random_device rd;
    pcg::pcg gen {rd(), rd()};

public:
    // --- constructors ---
    Engine(uint32_t P, uint32_t G) {
        this->P = P; this->G = G;
        this->scores.resize(P);
        this->idx.resize(P);
        this->pop.resize(P);

        // defaults scaled by P
        this->K = std::max<uint32_t>(1, P / 20);  // 5% elites
        this->R_base = std::max<uint32_t>(1, P / 10); // up to 10% randoms early; decays later
    }

    Engine(uint32_t P, uint32_t G,
           rand_t randomize,
           fit_t fit,
           crossbreed_t cross,
           mutate_t mutate,
           select_t select)
    : Engine(P, G) {
        this->randomize = std::move(randomize);
        this->fit = std::move(fit);
        this->cross = std::move(cross);
        this->mutate = std::move(mutate);
        this->select = std::move(select);
    }
    // -----------------

    // --- setters ---
    void set_fitness_function(const fit_t f){ fit = std::move(f); }
    void set_crossbreed_function(const crossbreed_t f){ cross = std::move(f); }
    void set_mutation_function(const mutate_t f){ mutate = std::move(f); }
    void set_selection_function(const select_t f){ select = std::move(f); }
    void set_randomize_function(const rand_t f){ randomize = std::move(f); }
    void set_distance_function(const dist_t f){ dist = std::move(f); }

    void set_population_size(uint32_t p){ P = p; }
    void set_total_generations(uint32_t g){ G = g; }
    void set_elites(uint32_t k){ K = k; }
    void set_base_random_immigrants(uint32_t r){ R_base = r; }
    void set_decay_random(bool d) { decay_R_base = d; }
    void set_reset_frequency(uint32_t f) { reset_freq = f; }
    void set_log_frequency(uint32_t f) { log_freq = f; }

    // TODO: Probably need to improve this
    void setSeed(uint64_t seed){ gen = {seed, seed}; }

    // --- getters ---
    uint32_t get_population_size() const { return P; }
    uint32_t get_total_generations() const { return G; }
    uint32_t get_current_gen() const { return cur_G; }
    const pop_t& get_population() const { return pop; }
    const score_t& get_scores() const { return scores; }
    const std::vector<size_t>& get_sorted_indices() const { return idx; }
    // ---------------

    // Reset (randomize) individual at index i
    void reset_specific(pcg::pcg pgen, size_t i) {
        pop[i] = std::move(randomize(pgen + i, *this));
    }

    // Cacluate fitness for all and maintain sorted indices
    void calc_scores() {
        #pragma omp parallel for
        for(size_t i = 0; i < P; ++i) {
            scores[i] = std::move(fit(pop[i]));
            idx[i] = i;
        }
        std::sort(std::execution::par, idx.begin(), idx.end(), [&](size_t a, size_t b){ return scores[a] < scores[b]; });
    }

    // Remove individuals too similar to current best; re-calculate scores
    void remove_similar() {
        #pragma omp parallel for
        for(size_t i = K; i < P; i++) {
            if(dist(pop[idx[0]], pop[idx[i]])) {
                reset_specific(gen + i, idx[i]);
            }
        }
        calc_scores();
    }

    // Randomize entire population
    void reset_pop() {
        #pragma omp parallel for
        for(size_t i = 0; i < P; ++i) {
            pop[i] = std::move(randomize(gen + i, *this));
        }
        gen = gen + P;
    }

    // Initialization
    void init() {
        reset_pop();
        calc_scores();
        cur_G = 0;
    }

    // One generation step
    void step() {
        // Calculate dynamic factors
        uint32_t R_dyn = R_base;
        if(decay_R_base) {
            double progress = static_cast<double>(cur_G) / std::max<uint32_t>(1, G - 1);
            double explore  = 1.0 - progress; // 1 → 0
            R_dyn  = std::max<uint32_t>(1, static_cast<uint32_t>(std::round(R_base * explore)));
        }

        pop_t new_pop(P);
        size_t pop_pos = 0;

        // Elitism
        const uint32_t elites = std::min<uint32_t>(K, P);
        #pragma omp parallel for
        for (size_t i = 0; i < elites; ++i) {
            new_pop[i] = pop[idx[i]];
        }
        pop_pos += elites;
        gen = gen + elites;

        // Random immigrants
        size_t immigrants = std::min<size_t>(R_dyn, P - pop_pos);
        #pragma omp parallel for
        for (size_t i = 0; i < immigrants; ++i) {
            new_pop[pop_pos + i] = randomize(gen + i, *this);
        }
        pop_pos += immigrants;
        gen = gen + immigrants;

        // Reproduction
        size_t parents = 2 * (P - pop_pos);
        std::vector<size_t> pairs = select(parents, gen, *this);
        gen = gen + parents;

        #pragma omp parallel for
        for (size_t par = 0; par < parents; par += 2) {
            auto child = cross(pop[pairs[par]], pop[pairs[par + 1]], gen + par, *this);
            mutate(child, gen + par + 1, *this);
            new_pop[pop_pos + par/2] = std::move(child);
        }

        pop = std::move(new_pop);
        calc_scores();
        cur_G++;
        gen = gen + G;
    }

    // Run GA
    pop_t simulate() {
        init();

        auto start = std::chrono::high_resolution_clock::now();
        for(uint32_t g = 0; g < G; ++g) {
            step();
            if(reset_freq > 0 && g % reset_freq == 0)
                remove_similar();
            if(log_freq > 0 && g % log_freq == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "Time: " << duration.count() << ", Generation: " << g << ", Best score: " << scores[idx[0]] << std::endl;
                start = end;
            }
        }

        // Return sorted list for easy usability
        pop_t sorted(P);
        for (size_t i = 0; i < P; ++i)
            sorted[i] = pop[idx[i]];
        return sorted;
    }
};


// ------------------------------------------
// --------- Standard Helper Functions

namespace crossbreed {
// Simple crossbreeding choosing gene with 'chance' probablity
template <typename gene_t = double, typename engine_t = Engine<>>
typename engine_t::crossbreed_t simple(double chance = 0.5,
                                       std::uniform_real_distribution<> U01 = std::uniform_real_distribution<>{0.0, 1.0}) {
    return [=](const std::vector<gene_t>& a,
               const std::vector<gene_t>& b,
               pcg::pcg gen,
               [[maybe_unused]] const engine_t& eng) mutable {

        const auto N = a.size();

        std::vector<gene_t> child(N);
        for(size_t i = 0; i < N; i++) {
            if(U01(gen) < chance)
                child[i] = a[i];
            else
                child[i] = b[i];
        }
        return child;
    };
}

// BLX-a crossover with dynamic alpha (decay)
template <typename gene_t = double, typename engine_t = Engine<>>
typename engine_t::crossbreed_t BLX_a(double alpha = 0.2, bool decay = false) {
    return [=](const std::vector<gene_t>& a,
               const std::vector<gene_t>& b,
               pcg::pcg gen,
               const engine_t& eng) {
        const auto N = a.size();
        double alpha_dyn = alpha;
        if(decay) {
            double progress = static_cast<double>(eng.get_current_gen()) /
                std::max<uint32_t>(1, eng.get_total_generations() - 1);
            double explore  = 1.0 - progress; // 1 → 0
            alpha_dyn = alpha * explore;
        }

        std::vector<gene_t> child(N);
        for(size_t i = 0; i < N; ++i) {
            double lo  = std::min(a[i], b[i]);
            double hi  = std::max(a[i], b[i]);
            double rng = hi - lo;
            lo -= alpha_dyn * rng;
            hi += alpha_dyn * rng;
            std::uniform_real_distribution<> dist(lo, hi);
            child[i] = dist(gen);
        }
        return child;
    };
}

// PMX cross; slices gene from B into middle of A
template <typename gene_t = double, typename param_t = std::vector<double>, typename engine_t = Engine<>>
typename engine_t::crossbreed_t PMX() {
    return [=](const param_t& a,
               const param_t& b,
               pcg::pcg gen,
               [[maybe_unused]] const engine_t& eng) {
        const int N = a.size();
        param_t child = [&] {
            if constexpr (std::is_same_v<param_t, std::vector<gene_t>>) {
                return param_t(N);   // empty vector of correct size
            } else {
                return param_t{};    // default-constructed array / object
            }
        }();

        std::uniform_int_distribution<> dist(0, N-1);
        int c1 = dist(gen), c2 = dist(gen);
        if(c1 > c2) std::swap(c1, c2);

        for(int i = c1; i <= c2; i++) child[i] = b[i];

        for(int i = 0; i < c1; i++) child[i] = a[i];
        for(int i = c2 + 1; i < N; i++) child[i] = a[i];

        return child;
    };
}

// PMX where genes cannot repeat; Assumes A and B contain same set of genes
template <typename gene_t = double, typename param_t = std::vector<double>, typename engine_t = Engine<>>
typename engine_t::crossbreed_t PMX_Uniques() {
    return [=](const param_t& a,
               const param_t& b,
               pcg::pcg gen,
               [[maybe_unused]] const engine_t& eng) {
        const int N = a.size();
        param_t child = [&] {
            if constexpr (std::is_same_v<param_t, std::vector<gene_t>>) {
                return param_t(N);   // empty vector of correct size
            } else {
                return param_t{};    // default-constructed array
            }
        }();

        std::uniform_int_distribution<> dist(0, N-1);
        int c1 = dist(gen), c2 = dist(gen);
        if(c1 > c2) std::swap(c1, c2);

        for(int i = c1; i <= c2; i++) child[i] = b[i];
        std::unordered_set<gene_t> used(child.begin()+c1, child.begin()+c2+1);

        int idx = 0;
        for(int i = 0; i < c1; i++) {
            while(used.count(b[idx])) idx++;
            child[i] = b[idx];
            idx++;
        }
        for(int i = c2 + 1; i < N; i++) {
            while(used.count(b[idx])) idx++;
            child[i] = b[idx];
            idx++;
        }

        return child;
    };
}

// Cyclic crossbreeding; Follow from gene to gene in cycles
template <typename gene_t = double, typename param_t = std::vector<double>, typename engine_t = Engine<>>
typename engine_t::crossbreed_t cycle() {
    return [=](const param_t& a,
               const param_t& b,
               [[maybe_unused]] pcg::pcg gen,
               [[maybe_unused]] const engine_t& eng) {
        const int N = a.size();
        param_t child = [&] {
            if constexpr (std::is_same_v<param_t, std::vector<gene_t>>) {
                return param_t(N);   // empty vector of correct size
            } else {
                return param_t{};    // default-constructed array
            }
        }();

        std::vector<bool> visited(N, false);
        std::unordered_map<gene_t, int> posB;
        for (int i=0;i<N;i++) posB[b[i]]=i;

        bool fromA = true;
        for (int start=0; start<N; start++) {
            if (visited[start]) continue;

            int idx=start;
            do {
                visited[idx]=true;
                child[idx] = fromA ? a[idx] : b[idx];
                idx = posB[a[idx]];
            } while (idx != start);

            fromA = !fromA;
        }
        return child;
    };
}
} // namespace crossbreed


namespace select {
// Tournament selection; T individuals compete
template <typename fitret_t = double, typename engine_t = Engine<>>
typename engine_t::select_t tournament(uint32_t T = 3) {
    return [=](size_t n,
               pcg::pcg gen,
               const engine_t& eng) mutable {
        const uint32_t P = eng.get_population_size();
        const auto& idx = eng.get_sorted_indices();
        std::uniform_int_distribution<> randint(0, P - 1);

        std::vector<size_t> pairs(n);
        for(size_t pr = 0; pr < n; pr++) {
            size_t best = randint(gen);
            for (size_t i = 1; i < T; ++i) {
                best = std::min<size_t>(best, randint(gen));
            }
            pairs[pr] = idx[best];
        }
        return pairs;
    };
}
} // namespace select


namespace mutate {
// Gaussian mutation with dynamic rate
template <typename param_t = std::vector<double>, typename engine_t = Engine<>>
typename engine_t::mutate_t simple(double mutate_min = 0.02,
                                   double mutate_max = 0.15,
                                   bool decay = true,
                                   std::uniform_real_distribution<> U01 = std::uniform_real_distribution<>{0.0, 1.0},
                                   std::normal_distribution<> gauss = std::normal_distribution<>{0.0, 0.30}) {
    return [=](param_t& a,
               pcg::pcg gen,
               const engine_t& eng) mutable {
        const auto N = a.size();
        double mutate_dyn = mutate_max;
        if(decay) {
            double progress = static_cast<double>(eng.get_current_gen()) / std::max<uint32_t>(1, eng.get_total_generations() - 1);
            double explore  = 1.0 - progress; // 1 → 0
            mutate_dyn  = mutate_min + (mutate_max - mutate_min) * explore;
        }

        for(size_t i = 0; i < N; ++i) {
            if (U01(gen) < mutate_dyn) {
                a[i] += gauss(gen);
            }
        }
    };
}

// Swap genes 'times' times
template <typename param_t = std::vector<double>, typename engine_t = Engine<>>
typename engine_t::mutate_t swap(size_t times) {
    return [=](param_t& a,
               pcg::pcg gen,
               [[maybe_unused]] const engine_t& eng) mutable {
        const int N = a.size();
        std::uniform_int_distribution<> dist(0, N-1);
        for(size_t x = 0; x < times; x++) {
            int i = dist(gen), j = dist(gen);
            std::swap(a[i], a[j]);
        }
    };
}

// Shuffle a portion of the DNA (continuous region)
template <typename param_t = std::vector<double>, typename engine_t = Engine<>>
typename engine_t::mutate_t scramble() {
    return [=](param_t& a,
               pcg::pcg gen,
               [[maybe_unused]] const engine_t& eng) mutable {
        const int N = a.size();
        std::uniform_int_distribution<> dist(0, N-1);
        int i = dist(gen), j = dist(gen);
        if(i > j) std::swap(i, j);
        std::shuffle(a.begin() + i, a.begin() + j, gen);
    };
}
} // namespace mutate


namespace randomize {
// Vector of size N, picked from a distribution
template <typename dist_t = std::uniform_real_distribution<>, typename gene_t = double, typename engine_t = Engine<>>
typename engine_t::rand_t randvector(uint32_t N, dist_t dist = dist_t{-1.0, 1.0}) {
    return [=](pcg::pcg gen, [[maybe_unused]] const engine_t& eng) mutable {
        std::vector<gene_t> x(N);
        for(size_t i = 0; i < N; ++i) x[i] = dist(gen);
        return x;
    };
}
} // namespace randomize


namespace fitness {
// Simple Mean Square Error - for sanity
template <typename gene_t = double, typename engine_t = Engine<>>
typename engine_t::fit_t simple() {
    return [=](const std::vector<gene_t>& p) mutable {
        double f = 0;
        for(double x : p)
        f += x * x;
        return f / (double) p.size();
    };
}
} // namespace fitness
} // namespace optimize
