#ifndef RNG_HPP
#define RNG_HPP

#include "common.hpp"

// 64-bit Linear Congruential Generator (Knuth MMIX constants)
__device__ inline float next_rand(uint64_t *seed)
{
    *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t val = (*seed >> 32);
    // Multiply by 1.0 / 2^32 to get [0, 1)
    return (float)val * 2.3283064365386963e-10f;
}

// Initialize seeds
__global__ void init_seed_kernel(uint64_t *seeds, int num_states, unsigned long long base_seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states)
    {
        uint64_t z = base_seed + tid;
        // SplitMix64 mixing
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        seeds[tid] = z;
    }
}

#endif // RNG_HPP