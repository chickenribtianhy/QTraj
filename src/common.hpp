#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuComplex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLOCK_SIZE 256
#define DEFAULT_TRAJECTORIES (1 << 10)
#define PRUNE_EVERY_N_GATES 100
#define PRUNE_PROB_THRESHOLD 0

// CUDA Error Checking Macro
#define checkCuda(result)                                                                                \
    {                                                                                                    \
        if ((result) != cudaSuccess)                                                                     \
        {                                                                                                \
            fprintf(stderr, "CUDA Error: %s - %s:%d\n", cudaGetErrorString(result), __FILE__, __LINE__); \
            exit(result);                                                                                \
        }                                                                                                \
    }

#endif // COMMON_HPP