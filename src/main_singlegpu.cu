//=============================================================================
//  trajectories_multi_gpu_debug.cu
//  - FIX: Solved Dangling Pointer Segfault (GroupedData ownership)
//  - DEBUG: Added granular checkpoints to trace execution flow
//=============================================================================

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <regex>
#include <cctype>
#include <chrono>
#include <limits>
#include <filesystem>

// --- OpenMP Includes ---
#include <omp.h>

// --- CUDA Includes ---
#include <cuda_runtime.h>
#include <cuComplex.h>

// --- Thrust Includes ---
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/binary_search.h>

#define BLOCK_SIZE 256
#define PRUNE_PROB_THRESHOLD 0

// --- DEBUG MACRO ---
#define DBG(msg) { \
    printf("[DEBUG][GPU %d] %s\n", omp_get_thread_num(), msg); \
    fflush(stdout); \
}

namespace fs = std::filesystem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define checkCuda(result)                                                                                \
    {                                                                                                    \
        if ((result) != cudaSuccess)                                                                     \
        {                                                                                                \
            fprintf(stderr, "CUDA Error: %s - %s:%d\n", cudaGetErrorString(result), __FILE__, __LINE__); \
            exit(result);                                                                                \
        }                                                                                                \
    }

// ===================================================================
// == LIGHTWEIGHT RANDOM NUMBER GENERATOR
// ===================================================================
__device__ inline float next_rand(uint64_t *seed)
{
    *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t val = (*seed >> 32);
    return (float)val * 2.3283064365386963e-10f;
}

__global__ void init_seed_kernel(uint64_t *seeds, int num_states, unsigned long long base_seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states)
    {
        uint64_t z = base_seed + tid;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        seeds[tid] = z;
    }
}

// ===================================================================
// == CUDA KERNELS
// ===================================================================

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

    double phase_update;
    if (do_flip)
        phase_update = (cur_state == 0) ? k01 : k10;
    else
        phase_update = (cur_state == 0) ? k00 : k11;

    phases[tid] += phase_update;
}

__global__ void cu3_kernel(uint8_t *data, uint64_t *seeds, double *phases,
                           int num_shots, int num_bytes_per_shot,
                           int control_index, int target_index,
                           double prob_flip,
                           double k00, double k01, double k10, double k11)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;

    // Check Control
    int c = get_qubit(row, control_index);
    if (c == 0)
        return;

    // Apply U3 logic to Target
    int cur_state = get_qubit(row, target_index);
    float r = next_rand(&seeds[tid]);

    int do_flip = (r < prob_flip);
    int new_state = cur_state ^ do_flip;
    set_qubit(row, target_index, new_state);

    double phase_update;
    if (do_flip)
        phase_update = (cur_state == 0) ? k01 : k10;
    else
        phase_update = (cur_state == 0) ? k00 : k11;

    phases[tid] += phase_update;
}

__global__ void cx_kernel(uint8_t *data, int num_shots, int num_bytes_per_shot,
                          int control_index, int target_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;
    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    int c = get_qubit(row, control_index);
    int t = get_qubit(row, target_index);
    set_qubit(row, target_index, c ^ t);
}

__global__ void ccx_kernel(uint8_t *data, int num_shots, int num_bytes_per_shot,
                           int c1_index, int c2_index, int target_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;
    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    int c1 = get_qubit(row, c1_index);
    int c2 = get_qubit(row, c2_index);
    int t = get_qubit(row, target_index);

    if (c1 == 1 && c2 == 1)
    {
        set_qubit(row, target_index, 1 - t);
    }
}

__global__ void swap_kernel(uint8_t *data, int num_shots, int num_bytes_per_shot,
                            int idx_a, int idx_b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    int a = get_qubit(row, idx_a);
    int b = get_qubit(row, idx_b);

    if (a != b)
    {
        set_qubit(row, idx_a, b);
        set_qubit(row, idx_b, a);
    }
}

__global__ void cswap_kernel(uint8_t *data, int num_shots, int num_bytes_per_shot,
                             int c_idx, int t1_idx, int t2_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;

    // Check Control
    int c = get_qubit(row, c_idx);
    if (c == 0)
        return;

    // Apply Swap Logic if Control is 1
    int a = get_qubit(row, t1_idx);
    int b = get_qubit(row, t2_idx);

    if (a != b)
    {
        set_qubit(row, t1_idx, b);
        set_qubit(row, t2_idx, a);
    }
}

// -------------------------------------------------------------------
// SORTING AND REDUCTION KERNELS
// -------------------------------------------------------------------

__global__ void extract_word_kernel(const uint8_t *data, const int *indices, uint64_t *out_words,
                                    int num_shots, int num_bytes_per_shot, int word_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    int shot_idx = indices[tid];
    const uint64_t *ptr = (const uint64_t *)(data + (size_t)shot_idx * num_bytes_per_shot);
    out_words[tid] = ptr[word_idx];
}

__global__ void mark_boundaries_kernel(const uint8_t *data, const int *indices, int *flags,
                                       int num_shots, int num_bytes_per_shot, int num_words)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    if (tid == 0)
    {
        flags[0] = 1;
        return;
    }

    int idx_curr = indices[tid];
    int idx_prev = indices[tid - 1];

    const uint64_t *row_curr = (const uint64_t *)(data + (size_t)idx_curr * num_bytes_per_shot);
    const uint64_t *row_prev = (const uint64_t *)(data + (size_t)idx_prev * num_bytes_per_shot);

    bool diff = false;
    for (int w = 0; w < num_words; ++w)
    {
        if (row_curr[w] != row_prev[w])
        {
            diff = true;
            break;
        }
    }
    flags[tid] = diff ? 1 : 0;
}

__global__ void gather_unique_states_kernel(const uint8_t *all_data, uint8_t *unique_data_out, 
                                            const int *unique_indices, 
                                            int num_unique, int num_bytes_per_shot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_unique)
        return;

    int src_shot_idx = unique_indices[tid];
    const uint8_t* src_ptr = all_data + (size_t)src_shot_idx * num_bytes_per_shot;
    uint8_t* dst_ptr = unique_data_out + (size_t)tid * num_bytes_per_shot;

    for(int i=0; i<num_bytes_per_shot; ++i) {
        dst_ptr[i] = src_ptr[i];
    }
}

__global__ void resample_wide_kernel(
    uint8_t *data,
    uint64_t *seeds,
    double *phases,
    int num_shots,
    int num_bytes_per_shot,
    const uint8_t *kept_states_flat,
    const double *kept_cdf,
    const double *kept_phases,
    int num_kept)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots)
        return;

    float u = next_rand(&seeds[tid]);
    double x = (u == 1.0f) ? 0.999999999999 : (double)u;

    int idx = 0;
    while (idx < num_kept - 1 && x > kept_cdf[idx])
        idx++;

    uint8_t *row = data + (size_t)tid * num_bytes_per_shot;
    const uint8_t *src = kept_states_flat + (size_t)idx * num_bytes_per_shot;

    int num_words = num_bytes_per_shot / 8;
    uint64_t *row64 = (uint64_t *)row;
    const uint64_t *src64 = (const uint64_t *)src;

    for (int i = 0; i < num_words; ++i)
    {
        row64[i] = src64[i];
    }

    phases[tid] = fmod(kept_phases[idx], 2.0 * M_PI);
}

// ===================================================================
// == QASM PARSING UTILS
// ===================================================================

static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
                                    { return !std::isspace(ch); }));
    return s;
}
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
                         { return !std::isspace(ch); })
                .base(),
            s.end());
    return s;
}
static inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

static double parse_angle(std::string s)
{
    trim(s);
    if (s.empty())
        return 0.0;
    size_t pi_pos = s.find("pi");
    if (pi_pos != std::string::npos)
    {
        std::string before_pi = s.substr(0, pi_pos);
        std::string after_pi = s.substr(pi_pos + 2);
        trim(before_pi);
        trim(after_pi);
        double multiplier = 1.0;
        double divisor = 1.0;
        if (!before_pi.empty())
        {
            if (before_pi == "+")
                multiplier = 1.0;
            else if (before_pi == "-")
                multiplier = -1.0;
            else
            {
                if (!before_pi.empty() && before_pi.back() == '*')
                    before_pi.pop_back();
                trim(before_pi);
                try
                {
                    multiplier = std::stod(before_pi);
                }
                catch (...)
                {
                    multiplier = 1.0;
                }
            }
        }
        if (!after_pi.empty() && after_pi.rfind('/', 0) == 0)
        {
            std::string denom_str = after_pi.substr(1);
            trim(denom_str);
            try
            {
                divisor = std::stod(denom_str);
                if (divisor == 0.0)
                    divisor = 1.0;
            }
            catch (...)
            {
                divisor = 1.0;
            }
        }
        return multiplier * M_PI / divisor;
    }
    try
    {
        return std::stod(s);
    }
    catch (...)
    {
        return 0.0;
    }
}

// ===================================================================
// == HOST CLASS
// ===================================================================

struct GateOp
{
    std::string type;
    std::vector<int> qubits;
    std::vector<double> params; // theta, phi, lambda
};

class Trajectories
{
public:
    struct GroupedData
    {
        std::vector<cuDoubleComplex> h_sum_w;
        std::vector<uint8_t> h_states_flat;
    };

    struct PhaseToComplexFunctor
    {
        __host__ __device__
            cuDoubleComplex
            operator()(const double &phase) const
        {
            return make_cuDoubleComplex(cos(phase), sin(phase));
        }
    };

    struct ReduceComplexAndIndex
    {
        __host__ __device__
            thrust::tuple<cuDoubleComplex, int>
            operator()(
                const thrust::tuple<cuDoubleComplex, int> &a,
                const thrust::tuple<cuDoubleComplex, int> &b) const
        {
            cuDoubleComplex sum_w = make_cuDoubleComplex(
                thrust::get<0>(a).x + thrust::get<0>(b).x,
                thrust::get<0>(a).y + thrust::get<0>(b).y);
            int rep_idx = (thrust::get<1>(a) < thrust::get<1>(b)) ? thrust::get<1>(a) : thrust::get<1>(b);
            return thrust::make_tuple(sum_w, rep_idx);
        }
    };

private:
    int num_qubits;
    int num_shots;
    int num_bytes_per_shot;
    int num_words_per_shot;

    uint8_t *d_data = nullptr;
    uint64_t *d_seeds = nullptr;
    double *d_phases = nullptr;

    dim3 getGridDim() const { return dim3((num_shots + BLOCK_SIZE - 1) / BLOCK_SIZE); }
    dim3 getBlockDim() const { return dim3(BLOCK_SIZE); }

    void sort_indices_by_state(thrust::device_vector<int> &d_indices)
    {
        thrust::device_vector<uint64_t> d_keys(num_shots);
        for (int w = 0; w < num_words_per_shot; ++w)
        {
            extract_word_kernel<<<getGridDim(), getBlockDim()>>>(
                d_data,
                thrust::raw_pointer_cast(d_indices.data()),
                thrust::raw_pointer_cast(d_keys.data()),
                num_shots, num_bytes_per_shot, w);
            checkCuda(cudaPeekAtLastError());
            thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());
        }
    }

public:
    GroupedData aggregate_results()
    {
        checkCuda(cudaDeviceSynchronize());
        
        // DBG("Starting Aggregation (Sorting)");
        thrust::device_vector<int> d_indices(num_shots);
        thrust::sequence(d_indices.begin(), d_indices.end());
        sort_indices_by_state(d_indices);

        // DBG("Marking Boundaries");
        thrust::device_vector<int> d_group_ids(num_shots);
        mark_boundaries_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data,
            thrust::raw_pointer_cast(d_indices.data()),
            thrust::raw_pointer_cast(d_group_ids.data()),
            num_shots, num_bytes_per_shot, num_words_per_shot);

        thrust::inclusive_scan(d_group_ids.begin(), d_group_ids.end(), d_group_ids.begin());

        auto perm_iter = thrust::make_permutation_iterator(
            thrust::device_pointer_cast(d_phases),
            d_indices.begin());

        auto complex_iter = thrust::make_transform_iterator(
            perm_iter,
            PhaseToComplexFunctor());

        auto values_begin = thrust::make_zip_iterator(
            thrust::make_tuple(complex_iter, d_indices.begin()));

        thrust::device_vector<cuDoubleComplex> d_out_weights(num_shots);
        thrust::device_vector<int> d_out_indices(num_shots);

        auto out_values_begin = thrust::make_zip_iterator(
            thrust::make_tuple(d_out_weights.begin(), d_out_indices.begin()));

        // DBG("Reducing by Key");
        auto end_pair = thrust::reduce_by_key(
            d_group_ids.begin(), d_group_ids.end(),
            values_begin,
            thrust::make_discard_iterator(),
            out_values_begin,
            thrust::equal_to<int>(),
            ReduceComplexAndIndex());

        int num_unique = (int)(end_pair.second - out_values_begin);
        // DBG("Found unique states");

        GroupedData res;
        res.h_sum_w.resize(num_unique);
        thrust::copy(d_out_weights.begin(), d_out_weights.begin() + num_unique, res.h_sum_w.begin());

        uint8_t *d_gathered_states = nullptr;
        checkCuda(cudaMalloc(&d_gathered_states, (size_t)num_unique * num_bytes_per_shot));

        int gridSize = (num_unique + 255) / 256;
        gather_unique_states_kernel<<<gridSize, 256>>>(
            d_data, d_gathered_states,
            thrust::raw_pointer_cast(d_out_indices.data()),
            num_unique, num_bytes_per_shot
        );
        checkCuda(cudaPeekAtLastError());

        res.h_states_flat.resize((size_t)num_unique * num_bytes_per_shot);
        checkCuda(cudaMemcpy(res.h_states_flat.data(), d_gathered_states, 
                             (size_t)num_unique * num_bytes_per_shot, cudaMemcpyDeviceToHost));

        checkCuda(cudaFree(d_gathered_states));
        // DBG("Aggregation Complete");

        return res;
    }

private:
    void prune_and_resample_threshold(double prob_threshold, int batch_id, int gate_step)
    {
        GroupedData gd = aggregate_results();
        size_t unique_states_count = gd.h_sum_w.size();
        if (unique_states_count == 0)
            return;

        double total_mag_sq = 0.0;
        double max_prob = 0.0;
        std::vector<double> mags_sq(unique_states_count);
        for (size_t i = 0; i < unique_states_count; ++i)
        {
            double mag = cuCabs(gd.h_sum_w[i]);
            double p = mag * mag;
            mags_sq[i] = p;
            total_mag_sq += p;
            if (p > max_prob)
                max_prob = p;
        }

        std::vector<size_t> kept_indices;
        for (size_t i = 0; i < mags_sq.size(); ++i)
        {
            double p = (total_mag_sq > 0) ? mags_sq[i] / total_mag_sq : 0.0;
            if (p >= prob_threshold)
                kept_indices.push_back(i);
        }

        if (kept_indices.empty())
        {
            auto it = std::max_element(mags_sq.begin(), mags_sq.end());
            kept_indices.push_back(std::distance(mags_sq.begin(), it));
        }

        double sum_kept = 0.0;
        for (size_t idx : kept_indices)
            sum_kept += mags_sq[idx];
        if (sum_kept <= 0.0)
            sum_kept = 1.0;

        std::vector<double> h_cdf;
        std::vector<uint8_t> h_kept_states_flat;
        std::vector<double> h_kept_phases;

        h_cdf.reserve(kept_indices.size());
        h_kept_states_flat.reserve(kept_indices.size() * num_bytes_per_shot);
        h_kept_phases.reserve(kept_indices.size());

        double acc = 0.0;
        for (size_t idx : kept_indices)
        {
            acc += mags_sq[idx] / sum_kept;
            h_cdf.push_back(acc);
            
            const uint8_t* ptr = &gd.h_states_flat[idx * num_bytes_per_shot];
            h_kept_states_flat.insert(h_kept_states_flat.end(), ptr, ptr + num_bytes_per_shot);

            double angle = atan2(gd.h_sum_w[idx].y, gd.h_sum_w[idx].x);
            h_kept_phases.push_back(angle);
        }
        h_cdf.back() = 1.0;

        uint8_t *d_kept_states = nullptr;
        double *d_kept_cdf = nullptr;
        double *d_kept_phases = nullptr;

        checkCuda(cudaMalloc(&d_kept_states, h_kept_states_flat.size()));
        checkCuda(cudaMalloc(&d_kept_cdf, h_cdf.size() * sizeof(double)));
        checkCuda(cudaMalloc(&d_kept_phases, h_kept_phases.size() * sizeof(double)));

        checkCuda(cudaMemcpy(d_kept_states, h_kept_states_flat.data(), h_kept_states_flat.size(), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_kept_cdf, h_cdf.data(), h_cdf.size() * sizeof(double), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_kept_phases, h_kept_phases.data(), h_kept_phases.size() * sizeof(double), cudaMemcpyHostToDevice));

        resample_wide_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, d_seeds, d_phases, num_shots, num_bytes_per_shot,
            d_kept_states, d_kept_cdf, d_kept_phases, (int)kept_indices.size());

        checkCuda(cudaFree(d_kept_states));
        checkCuda(cudaFree(d_kept_cdf));
        checkCuda(cudaFree(d_kept_phases));
    }

public:
    static std::vector<int> parse_qubits(const std::string &args_str, const std::map<std::string, int> &qubit_register_offsets)
    {
        std::vector<int> qubits;
        std::regex re("(\\w+)\\[(\\d+)\\]");
        std::sregex_iterator it(args_str.begin(), args_str.end(), re), end;
        while (it != end)
        {
            std::smatch m = *it;
            std::string reg_name = m.str(1);
            int reg_index = std::stoi(m.str(2));
            auto map_it = qubit_register_offsets.find(reg_name);
            if (map_it == qubit_register_offsets.end())
            {
                if (reg_name == "q" && qubit_register_offsets.size() == 1)
                    qubits.push_back(qubit_register_offsets.begin()->second + reg_index);
                else
                    throw std::runtime_error("Error: unknown qreg '" + reg_name + "'");
            }
            else
                qubits.push_back(map_it->second + reg_index);
            ++it;
        }
        return qubits;
    }

    static std::vector<GateOp> parse_qasm_to_ops(const std::string &filename, std::map<std::string, int> &register_map, int &out_num_qubits)
    {
        std::ifstream file(filename);
        if (!file.is_open())
            throw std::runtime_error("Error: Could not open QASM file.");

        register_map.clear();
        std::string line;
        std::regex re("qreg\\s+(\\w+)\\[(\\d+)\\]");
        int total = 0;
        bool found = false;

        std::vector<GateOp> ops;

        while (std::getline(file, line))
        {
            size_t comment_pos = line.find("//");
            if (comment_pos != std::string::npos)
                line = line.substr(0, comment_pos);
            trim(line);
            if (line.empty() || line.rfind("OPENQASM", 0) == 0 || line.rfind("include", 0) == 0)
                continue;

            std::smatch m;
            if (line.rfind("qreg", 0) == 0)
            {
                if (std::regex_search(line, m, re))
                {
                    register_map[m.str(1)] = total;
                    total += std::stoi(m.str(2));
                    found = true;
                }
                continue;
            }
            if (line.rfind("creg", 0) == 0 || line.rfind("measure", 0) == 0)
                continue;

            if (!line.empty() && line.back() == ';')
                line.pop_back();

            std::string command, params_str, args_str;
            size_t param_open = line.find('(');
            size_t space = line.find(' ');
            if (param_open != std::string::npos && (space == std::string::npos || param_open < space))
            {
                command = line.substr(0, param_open);
                size_t close = line.find(')');
                if (close != std::string::npos)
                    params_str = line.substr(param_open + 1, close - param_open - 1);
                size_t arg_s = line.find_first_not_of(' ', close + 1);
                if (arg_s != std::string::npos)
                    args_str = line.substr(arg_s);
            }
            else if (space != std::string::npos)
            {
                command = line.substr(0, space);
                args_str = line.substr(space + 1);
            }
            else
                command = line;

            trim(command);
            trim(params_str);
            trim(args_str);

            if (command.empty())
                continue;

            GateOp op;
            op.type = command;
            if (register_map.size() > 0)
                op.qubits = parse_qubits(args_str, register_map);

            if (!params_str.empty())
            {
                std::vector<std::string> tokens;
                std::stringstream ss(params_str);
                std::string token;
                while (std::getline(ss, token, ','))
                    op.params.push_back(parse_angle(token));
            }

            ops.push_back(op);
        }

        if (!found)
            throw std::runtime_error("Error: no qreg found.");
        out_num_qubits = total;
        return ops;
    }

    Trajectories(int n_qubits, int n_shots, uint64_t seed_offset) : num_qubits(n_qubits), num_shots(n_shots)
    {
        int bytes = (num_qubits + 7) / 8;
        num_words_per_shot = (bytes + 7) / 8;
        num_bytes_per_shot = num_words_per_shot * 8;

        size_t data_size = (size_t)num_shots * num_bytes_per_shot * sizeof(uint8_t);
        size_t seed_size = (size_t)num_shots * sizeof(uint64_t);
        size_t phase_size = (size_t)num_shots * sizeof(double);

        checkCuda(cudaMalloc(&d_data, data_size));
        checkCuda(cudaMalloc(&d_seeds, seed_size));
        checkCuda(cudaMalloc(&d_phases, phase_size));
        checkCuda(cudaMemset(d_data, 0, data_size));

        init_seed_kernel<<<getGridDim(), getBlockDim()>>>(d_seeds, num_shots, 1234ULL + seed_offset);
        checkCuda(cudaDeviceSynchronize());
        init_phase_kernel<<<getGridDim(), getBlockDim()>>>(d_phases, num_shots);
        checkCuda(cudaDeviceSynchronize());
    }

    ~Trajectories()
    {
        if (d_data)
            checkCuda(cudaFree(d_data));
        if (d_seeds)
            checkCuda(cudaFree(d_seeds));
        if (d_phases)
            checkCuda(cudaFree(d_phases));
    }

    // --- GATE IMPLEMENTATIONS ---

    void x(int q)
    {
        x_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, q);
        checkCuda(cudaPeekAtLastError());
    }

    void u3(int q, double theta, double phi, double lambda)
    {
        double half = theta / 2.0;
        double c = cos(half);
        double s = sin(half);
        double prob = fabs(s) / (fabs(c) + fabs(s));
        double k00 = (c >= 0) ? 0.0 : M_PI;
        double k01 = phi + ((s >= 0) ? 0.0 : M_PI);
        double k10 = lambda + ((s >= 0) ? M_PI : 0.0);
        double k11 = (phi + lambda) + ((c >= 0) ? 0.0 : M_PI);
        u3_kernel<<<getGridDim(), getBlockDim()>>>(d_data, d_seeds, d_phases, num_shots, num_bytes_per_shot, q, prob, k00, k01, k10, k11);
        checkCuda(cudaPeekAtLastError());
    }

    void cx(int c, int t)
    {
        cx_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, c, t);
        checkCuda(cudaPeekAtLastError());
    }

    void cu3(int c, int t, double theta, double phi, double lambda)
    {
        double half = theta / 2.0;
        double cos_v = cos(half);
        double sin_v = sin(half);
        double prob = fabs(sin_v) / (fabs(cos_v) + fabs(sin_v));

        double k00 = (cos_v >= 0) ? 0.0 : M_PI;
        double k01 = phi + ((sin_v >= 0) ? 0.0 : M_PI);
        double k10 = lambda + ((sin_v >= 0) ? M_PI : 0.0);
        double k11 = (phi + lambda) + ((cos_v >= 0) ? 0.0 : M_PI);

        cu3_kernel<<<getGridDim(), getBlockDim()>>>(d_data, d_seeds, d_phases, num_shots, num_bytes_per_shot, c, t, prob, k00, k01, k10, k11);
        checkCuda(cudaPeekAtLastError());
    }

    void ccx(int c1, int c2, int t)
    {
        ccx_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, c1, c2, t);
        checkCuda(cudaPeekAtLastError());
    }

    void swap(int a, int b)
    {
        swap_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, a, b);
        checkCuda(cudaPeekAtLastError());
    }

    void cswap(int c, int t1, int t2)
    {
        cswap_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, c, t1, t2);
        checkCuda(cudaPeekAtLastError());
    }

    // -- Derived Gates --
    void h(int q) { this->u3(q, M_PI / 2.0, 0.0, M_PI); }
    void t(int q) { this->u3(q, 0.0, 0.0, M_PI / 4.0); }
    void tdg(int q) { this->u3(q, 0.0, 0.0, -M_PI / 4.0); }
    void s(int q) { this->u3(q, 0.0, 0.0, M_PI / 2.0); }
    void sdg(int q) { this->u3(q, 0.0, 0.0, -M_PI / 2.0); }
    void sx(int q) { this->u3(q, M_PI / 2.0, -M_PI / 2.0, M_PI / 2.0); }

    void y(int q) { this->u3(q, M_PI, M_PI / 2.0, M_PI / 2.0); }
    void z(int q) { this->u3(q, 0.0, 0.0, M_PI); }

    void rx(int q, double theta) { this->u3(q, theta, -M_PI / 2.0, M_PI / 2.0); }
    void ry(int q, double theta) { this->u3(q, theta, 0.0, 0.0); }
    void rz(int q, double lambda) { this->u3(q, 0.0, 0.0, lambda); }

    void cy(int c, int t) { this->cu3(c, t, M_PI, M_PI / 2.0, M_PI / 2.0); }
    void cz(int c, int t) { this->cu3(c, t, 0.0, 0.0, M_PI); }
    void ch(int c, int t) { this->cu3(c, t, M_PI / 2.0, 0.0, M_PI); }
    void cp(int c, int t, double lambda) { this->cu3(c, t, 0.0, 0.0, lambda); }
    void crz(int c, int t, double lambda) { this->cu3(c, t, 0.0, 0.0, lambda); }

    void run_ops(const std::vector<GateOp> &ops, int batch_id)
    {
        const int PRUNE_INTERVAL = 100;
        int gate_count = 0;

        for (const auto &op : ops)
        {
            bool executed = true;
            // -- Single Qubit Non-Parametric --
            if (op.type == "x")
                x(op.qubits[0]);
            else if (op.type == "y")
                y(op.qubits[0]);
            else if (op.type == "z")
                z(op.qubits[0]);
            else if (op.type == "h")
                h(op.qubits[0]);
            else if (op.type == "s")
                s(op.qubits[0]);
            else if (op.type == "sdg")
                sdg(op.qubits[0]);
            else if (op.type == "t")
                t(op.qubits[0]);
            else if (op.type == "tdg")
                tdg(op.qubits[0]);
            else if (op.type == "sx")
                sx(op.qubits[0]);

            // -- Single Qubit Parametric --
            else if (op.type == "rx")
                rx(op.qubits[0], op.params[0]);
            else if (op.type == "ry")
                ry(op.qubits[0], op.params[0]);
            else if (op.type == "rz")
                rz(op.qubits[0], op.params[0]);
            else if ((op.type == "U" || op.type == "u" || op.type == "u3") && op.params.size() == 3)
                u3(op.qubits[0], op.params[0], op.params[1], op.params[2]);
            else if ((op.type == "u2") && op.params.size() == 2)
                u3(op.qubits[0], M_PI / 2.0, op.params[0], op.params[1]);
            else if ((op.type == "u1") && op.params.size() == 1)
                u3(op.qubits[0], 0.0, 0.0, op.params[0]);

            // -- Two Qubit Non-Parametric --
            else if (op.type == "cx")
                cx(op.qubits[0], op.qubits[1]);
            else if (op.type == "cy")
                cy(op.qubits[0], op.qubits[1]);
            else if (op.type == "cz")
                cz(op.qubits[0], op.qubits[1]);
            else if (op.type == "ch")
                ch(op.qubits[0], op.qubits[1]);
            else if (op.type == "swap")
                swap(op.qubits[0], op.qubits[1]);

            // -- Two Qubit Parametric --
            else if (op.type == "cu3" && op.params.size() == 3)
                cu3(op.qubits[0], op.qubits[1], op.params[0], op.params[1], op.params[2]);
            else if (op.type == "cp" || op.type == "cu1")
                cp(op.qubits[0], op.qubits[1], op.params[0]);
            else if (op.type == "crz")
                crz(op.qubits[0], op.qubits[1], op.params[0]);

            // -- Three Qubit --
            else if (op.type == "ccx")
                ccx(op.qubits[0], op.qubits[1], op.qubits[2]);
            else if (op.type == "cswap") 
                cswap(op.qubits[0], op.qubits[1], op.qubits[2]);

            else
                executed = false;

            if (executed)
            {
                gate_count++;
                if (gate_count % PRUNE_INTERVAL == 0)
                    prune_and_resample_threshold(PRUNE_PROB_THRESHOLD, batch_id, gate_count);
            }
        }
    }
};

int extract_n(const std::string &s)
{
    std::smatch m;
    std::regex re("_n(\\d+)");
    if (std::regex_search(s, m, re))
        return std::stoi(m.str(1));
    return 0;
}

static inline void print_complex_res(const cuDoubleComplex &z) { std::cout << "(" << std::fixed << std::setprecision(12) << z.x << (z.y >= 0 ? "+" : "") << z.y << "i)"; }

// --- CALCULATE BATCH SIZE DYNAMICALLY ---
long long calculate_max_batch_size(int n_qubits)
{
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if(err != cudaSuccess) {
        return 1 << 25; // Fallback
    }

    // 1. State Vector (Aligned)
    int bytes_per_state = ((n_qubits + 7) / 8); 
    bytes_per_state = ((bytes_per_state + 7) / 8) * 8; 

    // 2. Persistent (Seeds 8B + Phases 8B)
    int persistent_overhead = 16; 

    // 3. Transient Overhead (Reduction/Sorting)
    int transient_overhead = 32; 

    int bytes_per_shot = bytes_per_state + persistent_overhead + transient_overhead;

    // 4. Usage Target
    size_t usable_mem = (size_t)(free_mem * 0.95);

    long long max_shots = usable_mem / bytes_per_shot;
    
    // Safety clamp (min 100k shots)
    if(max_shots < 100000) max_shots = 100000;

    return max_shots;
}

int main(int argc, char *argv[])
{
    using clock = std::chrono::steady_clock;

    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <directory> <prefix> <output_file> <0/1> [optional: log_shots]\n";
        return 1;
    }

    std::string dir_path = argv[1];
    std::string prefix = argv[2];
    std::string out_file = argv[3];
    int show_results = std::stoi(argv[4]);
    int fixed_log = (argc == 6) ? std::stoi(argv[5]) : -1;

    // --- DETECT GPUS ---
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess || num_devices == 0) {
        std::cerr << "Error: No CUDA devices found.\n";
        return 1;
    }
    std::cout << ">>> Detected " << num_devices << " GPU(s).\n";

    if (!fs::exists(dir_path))
    {
        std::cerr << "Error: Directory not found: " << dir_path << "\n";
        return 1;
    }

    std::vector<fs::path> files;

    for (const auto &e : fs::directory_iterator(dir_path))
        if (e.path().extension() == ".qasm" && e.path().filename().string().find(prefix) == 0)
            files.push_back(e.path());

    std::sort(files.begin(), files.end(), [](const fs::path &a, const fs::path &b)
              { int na=extract_n(a.string()), nb=extract_n(b.string()); return na!=nb ? na<nb : a.string()<b.string(); });

    std::ofstream outfile(out_file);
    outfile << "Filename,Qubits,TotalRuntime(s)\n";

    for (const auto &fpath : files)
    {
        std::string fname = fpath.string(), sname = fpath.filename().string();
        std::cout << ">>> Benchmarking: " << sname << " ... \n";
        try
        {
            std::map<std::string, int> tm;
            int nq = 0;
            auto ops = Trajectories::parse_qasm_to_ops(fname, tm, nq);
            int num_bytes_per_shot = (((nq + 7) / 8 + 7) / 8) * 8; // Calculate this here for safety

            long long total_requested_shots = (fixed_log != -1) ? (1LL << fixed_log) : (1LL << std::min(30, nq + 5));

            std::map<std::vector<uint8_t>, cuDoubleComplex> global_results;
            auto t0 = clock::now();

            // --- MULTI-GPU EXECUTION (OPENMP) ---
            #pragma omp parallel num_threads(num_devices)
            {
                int dev_id = omp_get_thread_num();
                checkCuda(cudaSetDevice(dev_id));

                long long shots_per_device = total_requested_shots / num_devices;
                long long my_shots = shots_per_device;
                if (dev_id == num_devices - 1) {
                    my_shots += (total_requested_shots % num_devices);
                }

                if (my_shots > 0)
                {
                    long long max_gpu_shots = calculate_max_batch_size(nq);
                    int batch_size = std::min(max_gpu_shots, my_shots);
                    long long num_batches = (my_shots + batch_size - 1) / batch_size;

                    #pragma omp critical
                    {
                        std::cout << "    [GPU " << dev_id << "] Target Shots: " << my_shots 
                                  << " | Batches: " << num_batches 
                                  << " | Batch Size: " << batch_size << "\n";
                    }

                    // Local aggregation map
                    std::map<std::vector<uint8_t>, cuDoubleComplex> local_results;

                    for (long long b = 0; b < num_batches; ++b)
                    {
                        int current_shots = batch_size;
                        if (b == num_batches - 1)
                        {
                            current_shots = my_shots - (long long)b * batch_size;
                        }
                        if (current_shots <= 0) break;

                        uint64_t global_offset = ((uint64_t)dev_id * shots_per_device) + ((uint64_t)b * batch_size);

                        DBG("Init Circuit");
                        Trajectories circ(nq, current_shots, global_offset);
                        DBG("Running Ops");
                        circ.run_ops(ops, (int)b);
                        
                        DBG("Aggregating Results");
                        // FIX: Return the full GroupedData object to keep memory alive
                        auto gd = circ.aggregate_results();
                        DBG("Results Aggregated. Processing Map.");

                        size_t num_unique = gd.h_sum_w.size();
                        for (size_t i = 0; i < num_unique; ++i)
                        {
                            const uint8_t* ptr = &gd.h_states_flat[i * num_bytes_per_shot];
                            std::vector<uint8_t> key(ptr, ptr + num_bytes_per_shot);
                            cuDoubleComplex &w = local_results[key];
                            w = make_cuDoubleComplex(w.x + gd.h_sum_w[i].x, w.y + gd.h_sum_w[i].y);
                        }
                        DBG("Batch Done");
                    }

                    #pragma omp critical
                    {
                        DBG("Merging Global");
                        for (auto const& [state, val] : local_results) {
                            cuDoubleComplex &global_w = global_results[state];
                            global_w = make_cuDoubleComplex(global_w.x + val.x, global_w.y + val.y);
                        }
                    }
                }
            } // End OpenMP Parallel

            double dur = std::chrono::duration<double>(clock::now() - t0).count();

            if (show_results)
            {
                double total_mag_sq = 0.0;
                for (const auto &kv : global_results)
                {
                    double mag = cuCabs(kv.second);
                    total_mag_sq += mag * mag;
                }

                struct FinalRes
                {
                    std::vector<uint8_t> state;
                    double prob;
                    cuDoubleComplex avg_w;
                    std::string ket_str; 
                };

                std::vector<FinalRes> final_kept;
                double norm = sqrt(total_mag_sq);

                for (const auto &kv : global_results)
                {
                    double mag = cuCabs(kv.second);
                    double prob = (total_mag_sq > 0) ? (mag * mag) / total_mag_sq : 0.0;
                    
                    if (prob >= 1e-4)
                    {
                        FinalRes r;
                        r.state = kv.first;
                        r.prob = prob;
                        r.avg_w = make_cuDoubleComplex(kv.second.x / norm, kv.second.y / norm);

                        std::string s = "";
                        for (int q = 0; q < nq; ++q)
                        {
                            int byte = q / 8;
                            int bit = q % 8;
                            s += ((r.state[byte] >> bit) & 1) ? "1" : "0";
                        }
                        std::reverse(s.begin(), s.end());
                        r.ket_str = s;

                        final_kept.push_back(r);
                    }
                }

                std::sort(final_kept.begin(), final_kept.end(), [](const FinalRes &a, const FinalRes &b)
                          { return a.ket_str < b.ket_str; });

                std::cout << "\n    --- Results ---\n";
                for (const auto& item : final_kept)
                {
                    std::cout << "    |" << item.ket_str << ">: Prob=" << item.prob << " NormalizedW=";
                    print_complex_res(item.avg_w);
                    std::cout << "\n";
                }
            }
            std::cout << "    Done (" << dur << " s)\n";
            outfile << sname << "," << nq << "," << dur << "\n";
            outfile.flush();
        }
        catch (const std::exception &e)
        {
            std::cerr << "    [FAILED]: " << e.what() << "\n";
        }
    }
    return 0;
}