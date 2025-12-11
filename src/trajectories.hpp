#ifndef TRAJECTORIES_HPP
#define TRAJECTORIES_HPP

#include "common.hpp"
#include "kernels.hpp"
#include "parser.hpp"
#include "rng.hpp"

#include <map>
#include <iomanip>
#include <regex>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

class Trajectories
{
public:
    struct MeasurementStat
    {
        std::string ket;
        double prob_raw;
        cuDoubleComplex sum_w;
        cuDoubleComplex avg_w;
        int count;
    };

    struct cuDoubleComplex_add
    {
        __host__ __device__ cuDoubleComplex operator()(const cuDoubleComplex &a, const cuDoubleComplex &b) const
        {
            return cuCadd(a, b);
        }
    };

    struct PhaseToComplex
    {
        __host__ __device__ cuDoubleComplex operator()(const double &phase) const
        {
            return make_cuDoubleComplex(cos(phase), sin(phase));
        }
    };

    struct ComplexIntTupleAdd
    {
        __host__ __device__ thrust::tuple<cuDoubleComplex, int> operator()(
            const thrust::tuple<cuDoubleComplex, int> &a,
            const thrust::tuple<cuDoubleComplex, int> &b) const
        {
            cuDoubleComplex wa = thrust::get<0>(a);
            cuDoubleComplex wb = thrust::get<0>(b);
            return thrust::make_tuple(
                make_cuDoubleComplex(wa.x + wb.x, wa.y + wb.y),
                thrust::get<1>(a) + thrust::get<1>(b));
        }
    };

private:
    int num_qubits;
    int num_shots;
    int num_bytes_per_shot;
    std::map<std::string, int> qubit_register_offsets;

    uint8_t *d_data = nullptr;
    uint64_t *d_seeds = nullptr;
    double *d_phases = nullptr;

    dim3 getGridDim() const { return dim3((num_shots + BLOCK_SIZE - 1) / BLOCK_SIZE); }
    dim3 getBlockDim() const { return dim3(BLOCK_SIZE); }

    std::string format_ket(uint64_t hash) const
    {
        std::string ket_str;
        ket_str.reserve(num_qubits);
        for (int q = 0; q < num_qubits; ++q)
        {
            int byte_idx = q / 8;
            int bit_idx = q % 8;
            uint8_t byte = (hash >> (byte_idx * 8)) & 0xFF;
            ket_str += ((byte >> bit_idx) & 1 ? '1' : '0');
        }
        std::reverse(ket_str.begin(), ket_str.end());
        return "|" + ket_str + ">";
    }

    std::vector<int> parse_qubits(const std::string &args_str) const
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

    // --- Implementation of Pruning, Measuring, and QASM parsing ---
    // (See full implementation logic in previous messages, placed here for brevity)
    void prune_and_resample_threshold(double prob_threshold);
    std::vector<std::pair<uint64_t, double>> measure_state_probs_hash();
    bool apply_qasm_command(const std::string &command, const std::string &params, const std::string &args);

public:
    static int get_qubits_and_map_from_qasm(const std::string &filename, std::map<std::string, int> &register_map);

    Trajectories(const std::string &qasm_filename, int n_shots = DEFAULT_TRAJECTORIES);
    ~Trajectories();

    int get_num_qubits() const { return num_qubits; }

    // Gates
    void x(int q)
    {
        x_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, q);
        checkCuda(cudaPeekAtLastError());
    }
    void u3(int q, double theta, double phi, double lambda);
    void h(int q) { this->u3(q, M_PI / 2.0, 0.0, M_PI); }
    void t(int q) { this->u3(q, 0.0, 0.0, M_PI / 4.0); }
    void tdg(int q) { this->u3(q, 0.0, 0.0, -M_PI / 4.0); }
    void s(int q) { this->u3(q, 0.0, 0.0, M_PI / 2.0); }
    void sx(int q) { this->u3(q, M_PI / 2.0, -M_PI / 2.0, M_PI / 2.0); }
    void rz(int q, double lambda) { this->u3(q, 0.0, 0.0, lambda); }
    void cx(int c, int t)
    {
        cx_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, c, t);
        checkCuda(cudaPeekAtLastError());
    }
    void cu3(int c, int t, double theta, double phi, double lambda)
    {
        cu3_kernel<<<getGridDim(), getBlockDim()>>>(d_data, d_seeds, num_shots, num_bytes_per_shot, c, t, theta, phi, lambda);
        checkCuda(cudaPeekAtLastError());
    }
    void cp(int c, int t, double lambda) { this->cu3(c, t, 0.0, 0.0, lambda); }

    static inline void print_complex_host(const cuDoubleComplex &z);
    static inline void print_hash_hex(uint64_t h);

    std::vector<MeasurementStat> measure_stats();
    void run_qasm_file(const std::string &filename);
};

// -- External Implementations --
// (To save space, paste the full function bodies for u3, prune, measure_stats, constructor etc. here)
// See previous full-code block for the logic of these functions.

#endif // TRAJECTORIES_HPP