#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "common.hpp"
#include "rng.hpp"

// --- Device Helpers ---

__device__ inline int get_qubit(const uint8_t *shot_row, int qubit_index)
{
    int byte_index = qubit_index / 8;
    int bit_index = qubit_index % 8;
    uint8_t mask = (uint8_t)(1u << bit_index);
    return (shot_row[byte_index] & mask) >> bit_index;
}

__device__ inline void set_qubit(uint8_t *shot_row, int qubit_index, int new_state)
{
    int byte_index = qubit_index / 8;
    int bit_index = qubit_index % 8;
    uint8_t mask = (uint8_t)(1u << bit_index);
    uint8_t byte = shot_row[byte_index];
    byte = (uint8_t)((byte & ~mask) | ((new_state & 1) << bit_index));
    shot_row[byte_index] = byte;
}

// --- Kernels ---

__global__ void init_phase_kernel(double *phases, int num_states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states)
        phases[tid] = 0.0;
}

__global__ void x_kernel(uint8_t *data, int num_shots, int num_bytes_per_shot, int qubit_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;
    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    int cur = get_qubit(row, qubit_index);
    set_qubit(row, qubit_index, 1 - cur);
}

__global__ void u3_kernel(uint8_t *data, uint64_t *seeds, double *phases,
                          int num_shots, int num_bytes_per_shot, int qubit_index,
                          double prob_flip,
                          double k00, double k01, double k10, double k11)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    int cur_state = get_qubit(row, qubit_index);

    float r = next_rand(&seeds[tid]);
    int do_flip = (r < prob_flip);
    int new_state = cur_state ^ do_flip;
    set_qubit(row, qubit_index, new_state);

    double phase_update = (do_flip) ? ((cur_state == 0) ? k01 : k10)
                                    : ((cur_state == 0) ? k00 : k11);
    phases[tid] += phase_update;
}

__global__ void cx_kernel(uint8_t *data, int num_shots, int num_bytes_per_shot, int control_index, int target_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;
    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    int c = get_qubit(row, control_index);
    int t = get_qubit(row, target_index);
    set_qubit(row, target_index, c ^ t);
}

__global__ void cu3_kernel(uint8_t *data, uint64_t *seeds,
                           int num_shots, int num_bytes_per_shot,
                           int control_index, int target_index,
                           double theta, double phi, double lambda)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;
    if (theta == 0.0)
        return;

    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    int c = get_qubit(row, control_index);
    if (c == 0)
        return;

    double cos_theta_half = cos(theta / 2.0);
    double sin_theta_half = sin(theta / 2.0);
    double prob0_to_1 = abs(sin_theta_half) / (abs(cos_theta_half) + abs(sin_theta_half));
    double prob1_to_1 = abs(cos_theta_half) / (abs(cos_theta_half) + abs(sin_theta_half));

    int t = get_qubit(row, target_index);
    double prob_to_be_1 = (t == 1) ? prob1_to_1 : prob0_to_1;

    float r = next_rand(&seeds[tid]);
    int new_state = (r < prob_to_be_1) ? 1 : 0;
    set_qubit(row, target_index, new_state);
}

__global__ void pack_to_uint64_kernel(uint8_t *data, uint64_t *hashes, int num_shots, int num_bytes_per_shot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;
    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    uint64_t h = 0;
    for (int i = 0; i < num_bytes_per_shot; ++i)
    {
        h |= (uint64_t)row[i] << (i * 8);
    }
    hashes[tid] = h;
}

__global__ void resample_from_kept_states_kernel(
    uint8_t *data, uint64_t *seeds, double *phases,
    int num_shots, int num_bytes_per_shot,
    const uint64_t *kept_hashes, const double *kept_cdf, int num_kept)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    float u = next_rand(&seeds[tid]);
    double x = (u == 1.0f) ? 0.999999999999 : (double)u;

    int idx = 0;
    while (idx < num_kept - 1 && x > kept_cdf[idx])
        idx++;

    uint64_t h = kept_hashes[idx];
    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    for (int i = 0; i < num_bytes_per_shot; ++i)
    {
        row[i] = (uint8_t)((h >> (i * 8)) & 0xFF);
    }
    phases[tid] = 0.0;
}

#endif // KERNELS_HPP