#include <cstdint>

namespace pcg {

struct pcg {
    using result_type = std::uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }

    uint64_t state;
    uint64_t inc;

    uint32_t operator() () {
        uint64_t oldstate = this->state;
        this->state = oldstate * 6364136223846793005ULL + (this->inc|1);
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    pcg operator+ (uint64_t i) {
        return {state + i, inc + i};
    }
};

}
