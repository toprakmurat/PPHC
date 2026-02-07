#include "math/ntt_core.h"
#include <stdexcept>
#include <bit>
#include <algorithm>

namespace securepulse::math {

NTTCore::NTTCore(const Parameters& params) : params_(params) {
    if (!std::has_single_bit(params.ring_dimension)) {
        throw std::invalid_argument("Ring dimension must be a power of 2");
    }
    precompute_twiddles();
}

NTTCore::~NTTCore() = default;

NTTCore::NTTCore(NTTCore&&) noexcept = default;
NTTCore& NTTCore::operator=(NTTCore&&) noexcept = default;

void NTTCore::precompute_twiddles() {
    // TODO: Implement twiddle factor precomputation
    // Powers of root_of_unity mod modulus
    twiddle_factors_.resize(params_.ring_dimension);
    inv_twiddle_factors_.resize(params_.ring_dimension);
}

void NTTCore::forward(std::span<uint64_t> data) const {
    if (data.size() != params_.ring_dimension) {
        throw std::invalid_argument("Data size must match ring dimension");
    }
    ct_butterfly(data, twiddle_factors_);
}

void NTTCore::inverse(std::span<uint64_t> data) const {
    if (data.size() != params_.ring_dimension) {
        throw std::invalid_argument("Data size must match ring dimension");
    }
    gs_butterfly(data, inv_twiddle_factors_);
    
    // Scale by n^{-1}
    for (auto& val : data) {
        val = (static_cast<__uint128_t>(val) * n_inverse_) % params_.modulus;
    }
}

std::vector<uint64_t> NTTCore::multiply(
    std::span<const uint64_t> a,
    std::span<const uint64_t> b
) const {
    std::vector<uint64_t> result(params_.ring_dimension);
    std::vector<uint64_t> a_ntt(a.begin(), a.end());
    std::vector<uint64_t> b_ntt(b.begin(), b.end());
    
    forward(a_ntt);
    forward(b_ntt);
    
    for (size_t i = 0; i < params_.ring_dimension; ++i) {
        result[i] = (static_cast<__uint128_t>(a_ntt[i]) * b_ntt[i]) % params_.modulus;
    }
    
    inverse(result);
    return result;
}

void NTTCore::ct_butterfly(
    std::span<uint64_t> data,
    const std::vector<uint64_t>& twiddles
) const {
    // TODO: Cooley-Tukey radix-2 DIT implementation
    (void)data;
    (void)twiddles;
}

void NTTCore::gs_butterfly(
    std::span<uint64_t> data,
    const std::vector<uint64_t>& twiddles
) const {
    // TODO: Gentleman-Sande radix-2 DIF implementation
    (void)data;
    (void)twiddles;
}

uint64_t NTTCore::barrett_reduce(
    __uint128_t x,
    uint64_t modulus,
    uint64_t barrett_factor
) noexcept {
    uint64_t q = static_cast<uint64_t>((x * barrett_factor) >> 64);
    uint64_t r = static_cast<uint64_t>(x) - q * modulus;
    return r >= modulus ? r - modulus : r;
}

uint64_t NTTCore::montgomery_mult(
    uint64_t a,
    uint64_t b,
    uint64_t modulus,
    uint64_t mont_factor
) noexcept {
    __uint128_t t = static_cast<__uint128_t>(a) * b;
    uint64_t m = static_cast<uint64_t>(t) * mont_factor;
    __uint128_t u = t + static_cast<__uint128_t>(m) * modulus;
    uint64_t result = static_cast<uint64_t>(u >> 64);
    return result >= modulus ? result - modulus : result;
}

std::unique_ptr<NTTCore> NTTAccelerator::create(
    const NTTCore::Parameters& params,
    const Config& config
) {
    (void)config;
    return std::make_unique<NTTCore>(params);
}

bool NTTAccelerator::is_backend_available(BackendType backend) {
    switch (backend) {
        case BackendType::CPU_SCALAR:
            return true;
        default:
            return false;
    }
}

} // namespace securepulse::math
