# PPHC

### Privacy-Focused Health Risk Analysis with Fully Homomorphic Encryption

---

## Overview

**PPHC** is a Hardware-Software Co-design framework for privacy-preserving health analysis using **Fully Homomorphic Encryption (FHE)**. The system enables secure computation on encrypted medical data without ever exposing sensitive patient information—achieving true **"Encryption in Use"** capability.

### Zero-Trust Architecture

This project implements a Zero-Trust security model where:
- Patient data is encrypted on the client side and **never decrypted** on the server
- The server performs all computations directly on encrypted ciphertexts
- Only the data owner holds the decryption keys
- Results are returned encrypted and decrypted locally by the patient

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PPHC                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
│  │   Models    │───▶│   Core Engine    │◀───│  Hardware Accelerator │  │
│  │  (Python)   │    │     (C++20)      │    │   (Verilog/FPGA)      │  │
│  └─────────────┘    └────────┬─────────┘    └───────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│                     ┌──────────────────┐                               │
│                     │  Backend Server  │                               │
│                     │     (C++20)      │                               │
│                     └──────────────────┘                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Fully Homomorphic Encryption**: Compute on encrypted data using CKKS/BFV/BGV schemes
- **Hardware Acceleration**: FPGA-based NTT and modular arithmetic accelerators
- **Quantized ML Models**: 8-bit quantized inference for FHE compatibility
- **Multi-Library Support**: Benchmarks across OpenFHE, Microsoft SEAL, and Concrete

---

## Project Structure

```
PPHC/
├── core/                   # C++ FHE math kernels
│   ├── include/
│   │   └── math/
│   │       └── ntt_core.h  # Number Theoretic Transform
│   └── src/
├── backend/                # Server application (C++)
├── models/                 # Python ML training & quantization
├── hardware/               # FPGA/ASIC Verilog designs
│   └── fpga/
├── benchmarks/             # Performance comparisons
├── docker/                 # Containerization
└── CMakeLists.txt          # Build configuration
```

---

## Technical Stack

| Layer      | Technology                              |
|------------|-----------------------------------------|
| Core       | C++20, OpenFHE, Microsoft SEAL          |
| Models     | Python 3.10+, scikit-learn, NumPy       |
| Hardware   | Verilog, SystemVerilog, Vivado          |
| Build      | CMake 3.20+, Ninja                      |
| Testing    | GoogleTest, pytest, Verilator           |

---

## Building

### Prerequisites

- CMake >= 3.20
- C++20 compatible compiler (GCC 11+, Clang 14+, MSVC 2022)
- Python >= 3.10
- OpenFHE or Microsoft SEAL (optional)

### Compilation

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

---

## FHE Performance Bottlenecks

The primary computational bottlenecks in FHE addressed by this project:

1. **Number Theoretic Transform (NTT)** - O(n log n) polynomial multiplication
2. **Modular Multiplication** - Barrett/Montgomery reduction
3. **Key Switching** - Relinearization and rotation

Hardware acceleration targets these operations for 10-100x speedup.

---

## License

This project is for academic and research purposes.

---

## References

- Cheon, J. H., et al. "Homomorphic Encryption for Arithmetic of Approximate Numbers" (CKKS)
- Brakerski, Z., et al. "Fully Homomorphic Encryption without Bootstrapping" (BGV/BFV)
- OpenFHE: https://www.openfhe.org/
- Microsoft SEAL: https://github.com/microsoft/SEAL
