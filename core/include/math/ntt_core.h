#ifndef SECUREPULSE_MATH_NTT_CORE_H
#define SECUREPULSE_MATH_NTT_CORE_H

#include <cstdint>
#include <vector>
#include <span>
#include <memory>

namespace securepulse::math {

/**
 * @brief Number Theoretic Transform (NTT) for polynomial multiplication
 * 
 * The NTT is a critical performance bottleneck in FHE schemes.
 * This implementation targets:
 *   - Negacyclic NTT for ring Z_q[X]/(X^n + 1)
 *   - Constant-geometry butterfly operations for hardware acceleration
 *   - AVX-512 / ARM NEON vectorization paths
 */
class NTTCore {
public:
    struct Parameters {
        uint64_t modulus;           // Prime modulus q
        uint64_t root_of_unity;     // Primitive 2n-th root of unity
        size_t   ring_dimension;    // Polynomial degree n (power of 2)
    };

    explicit NTTCore(const Parameters& params);
    ~NTTCore();

    NTTCore(const NTTCore&) = delete;
    NTTCore& operator=(const NTTCore&) = delete;
    NTTCore(NTTCore&&) noexcept;
    NTTCore& operator=(NTTCore&&) noexcept;

    /**
     * @brief Forward NTT: coefficient to evaluation domain
     * @param[in,out] data Polynomial coefficients (modified in-place)
     */
    void forward(std::span<uint64_t> data) const;

    /**
     * @brief Inverse NTT: evaluation to coefficient domain
     * @param[in,out] data Polynomial evaluations (modified in-place)
     */
    void inverse(std::span<uint64_t> data) const;

    /**
     * @brief Negacyclic polynomial multiplication via NTT
     * @param a First polynomial (coefficient form)
     * @param b Second polynomial (coefficient form)
     * @return Product polynomial (coefficient form)
     */
    [[nodiscard]] std::vector<uint64_t> multiply(
        std::span<const uint64_t> a,
        std::span<const uint64_t> b
    ) const;

    [[nodiscard]] size_t ring_dimension() const noexcept { return params_.ring_dimension; }
    [[nodiscard]] uint64_t modulus() const noexcept { return params_.modulus; }

private:
    Parameters params_;

    std::vector<uint64_t> twiddle_factors_;
    std::vector<uint64_t> inv_twiddle_factors_;
    uint64_t n_inverse_;  // Multiplicative inverse of n mod q

    void precompute_twiddles();

    /**
     * @brief Cooley-Tukey butterfly: constant-geometry NTT
     */
    void ct_butterfly(
        std::span<uint64_t> data,
        const std::vector<uint64_t>& twiddles
    ) const;

    /**
     * @brief Gentleman-Sande butterfly: inverse NTT
     */
    void gs_butterfly(
        std::span<uint64_t> data,
        const std::vector<uint64_t>& twiddles
    ) const;

    /**
     * @brief Barrett reduction: modular reduction without division
     */
    [[nodiscard]] static uint64_t barrett_reduce(
        __uint128_t x,
        uint64_t modulus,
        uint64_t barrett_factor
    ) noexcept;

    /**
     * @brief Montgomery multiplication: modular multiplication
     */
    [[nodiscard]] static uint64_t montgomery_mult(
        uint64_t a,
        uint64_t b,
        uint64_t modulus,
        uint64_t mont_factor
    ) noexcept;
};

/**
 * @brief Hardware-accelerated NTT interface (FPGA/ASIC offload)
 */
class NTTAccelerator {
public:
    enum class BackendType {
        CPU_SCALAR,
        CPU_AVX512,
        CPU_NEON,
        FPGA_PCIE,
        ASIC
    };

    struct Config {
        BackendType backend = BackendType::CPU_SCALAR;
        size_t max_ring_dimension = 65536;
        bool enable_batch_mode = true;
    };

    static std::unique_ptr<NTTCore> create(
        const NTTCore::Parameters& params,
        const Config& config = {}
    );

    static bool is_backend_available(BackendType backend);
};

} // namespace securepulse::math

#endif // SECUREPULSE_MATH_NTT_CORE_H
