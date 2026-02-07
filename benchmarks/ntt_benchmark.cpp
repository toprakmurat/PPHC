#include <benchmark/benchmark.h>
#include "math/ntt_core.h"
#include <random>

using namespace securepulse::math;

static void BM_NTT_Forward(benchmark::State& state) {
    const size_t n = state.range(0);
    NTTCore::Parameters params{
        .modulus = 0xFFFFFFFF00000001ULL,
        .root_of_unity = 0x0,
        .ring_dimension = n
    };
    
    NTTCore ntt(params);
    std::vector<uint64_t> data(n);
    std::mt19937_64 rng(42);
    for (auto& v : data) v = rng() % params.modulus;
    
    for (auto _ : state) {
        ntt.forward(data);
        benchmark::DoNotOptimize(data);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_NTT_Forward)->RangeMultiplier(2)->Range(1024, 65536);
