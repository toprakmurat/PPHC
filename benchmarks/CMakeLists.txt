cmake_minimum_required(VERSION 3.20)

include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(bench_ntt ntt_benchmark.cpp)
target_link_libraries(bench_ntt PRIVATE
    securepulse_core
    benchmark::benchmark
    benchmark::benchmark_main
)
