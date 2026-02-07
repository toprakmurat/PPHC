cmake_minimum_required(VERSION 3.20)

project(SecurePulse-FHE
    VERSION 0.1.0
    DESCRIPTION "Hardware-Software Co-design for Privacy-Preserving Health Analysis using FHE"
    LANGUAGES CXX
)

# ─────────────────────────────────────────────────────────────────────────────
# C++ Standard Configuration
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────────────────────────────────────────────────────────
# Build Type Configuration
# ─────────────────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# ─────────────────────────────────────────────────────────────────────────────
# Project Options
# ─────────────────────────────────────────────────────────────────────────────
option(SECUREPULSE_BUILD_TESTS "Build unit tests" ON)
option(SECUREPULSE_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(SECUREPULSE_USE_OPENFHE "Use OpenFHE library" OFF)
option(SECUREPULSE_USE_SEAL "Use Microsoft SEAL library" OFF)

# ─────────────────────────────────────────────────────────────────────────────
# External Dependencies
# ─────────────────────────────────────────────────────────────────────────────

# OpenFHE (optional)
if(SECUREPULSE_USE_OPENFHE)
    find_package(OpenFHE REQUIRED)
    if(OpenFHE_FOUND)
        message(STATUS "OpenFHE found: ${OpenFHE_VERSION}")
        add_definitions(-DUSE_OPENFHE)
    endif()
endif()

# Microsoft SEAL (optional)
if(SECUREPULSE_USE_SEAL)
    find_package(SEAL 4.0 REQUIRED)
    if(SEAL_FOUND)
        message(STATUS "Microsoft SEAL found: ${SEAL_VERSION}")
        add_definitions(-DUSE_SEAL)
    endif()
endif()

# GoogleTest for unit testing
if(SECUREPULSE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Include Directories
# ─────────────────────────────────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/core/include
)

# ─────────────────────────────────────────────────────────────────────────────
# Core Library
# ─────────────────────────────────────────────────────────────────────────────
file(GLOB_RECURSE CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/core/src/*.cpp
)

add_library(securepulse_core STATIC
    ${CORE_SOURCES}
)

target_include_directories(securepulse_core PUBLIC
    ${CMAKE_SOURCE_DIR}/core/include
)

if(SECUREPULSE_USE_OPENFHE AND OpenFHE_FOUND)
    target_link_libraries(securepulse_core PUBLIC OpenFHE::OpenFHE)
endif()

if(SECUREPULSE_USE_SEAL AND SEAL_FOUND)
    target_link_libraries(securepulse_core PUBLIC SEAL::seal)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Backend Server
# ─────────────────────────────────────────────────────────────────────────────
add_executable(securepulse_server
    ${CMAKE_SOURCE_DIR}/backend/server.cpp
)

target_link_libraries(securepulse_server PRIVATE
    securepulse_core
)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
if(SECUREPULSE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Installation
# ─────────────────────────────────────────────────────────────────────────────
install(TARGETS securepulse_server securepulse_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY core/include/
    DESTINATION include
)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "SecurePulse-FHE Configuration Summary")
message(STATUS "======================================")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenFHE:       ${SECUREPULSE_USE_OPENFHE}")
message(STATUS "  SEAL:          ${SECUREPULSE_USE_SEAL}")
message(STATUS "  Tests:         ${SECUREPULSE_BUILD_TESTS}")
message(STATUS "  Benchmarks:    ${SECUREPULSE_BUILD_BENCHMARKS}")
message(STATUS "")
