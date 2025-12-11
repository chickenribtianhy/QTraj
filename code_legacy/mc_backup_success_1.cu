//===============================================================
//  trajectories_qasm_cuda.cu (single-file build)
//  - Parses a subset of OpenQASM 2.0
//  - Executes gates on GPU trajectories
//  - Every 100 executed gates -> measure -> prune tiny states (p < THRESH) -> resample/reset
//  - Final measurement: discards tiny p, renormalizes kept, and prints BOTH probability + avg(weight)
//    where avg(weight) = (summed_weight_for_state / count_of_shots_in_state)
//===============================================================

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

// --- CUDA Includes ---
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>

// --- Thrust Includes (for measure()) ---
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_ptr.h>

#define BLOCK_SIZE 256
#define DEFAULT_TRAJECTORIES (1 << 10)
#define PRUNE_EVERY_N_GATES 100
#define PRUNE_PROB_THRESHOLD 0
namespace fs = std::filesystem;

// --- CUDA Error Checking Macro ---
#define checkCuda(result) { \
    if ((result) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s - %s:%d\n", cudaGetErrorString(result), __FILE__, __LINE__); \
        exit(result); \
    } \
}

// ===================================================================
// == CUDA KERNELS & DEVICE FUNCTIONS
// ===================================================================

__device__ inline int get_qubit(const uint8_t* shot_row, int qubit_index) {
    int byte_index = qubit_index / 8;
    int bit_index  = qubit_index % 8;
    uint8_t mask   = (uint8_t)(1u << bit_index);
    return (shot_row[byte_index] & mask) >> bit_index;
}

__device__ inline void set_qubit(uint8_t* shot_row, int qubit_index, int new_state) {
    int byte_index = qubit_index / 8;
    int bit_index  = qubit_index % 8;
    uint8_t mask   = (uint8_t)(1u << bit_index);

    uint8_t byte = shot_row[byte_index];
    byte = (uint8_t)((byte & ~mask) | ((new_state & 1) << bit_index));
    shot_row[byte_index] = byte;
}

__global__ void init_curand_kernel(curandState* states, int num_states, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

__global__ void init_weight_kernel(cuDoubleComplex* weights, int num_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states) {
        weights[tid] = make_cuDoubleComplex(1.0, 0.0);
    }
}

__global__ void x_kernel(uint8_t* data, int num_shots, int num_bytes_per_shot, int qubit_index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* row = data + (size_t)tid * num_bytes_per_shot;
    int cur = get_qubit(row, qubit_index);
    set_qubit(row, qubit_index, 1 - cur);
}

__global__ void u3_kernel(uint8_t* data, curandState* rand_states, cuDoubleComplex* weights,
                         int num_shots, int num_bytes_per_shot,
                         int qubit_index, double theta, double phi, double lambda) {
                        //  cuDoubleComplex global_phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* row = data + (size_t)tid * num_bytes_per_shot;

    double cos_theta_half = cos(theta / 2.0);
    double sin_theta_half = sin(theta / 2.0);

    // Probabilistic branching for basis state bit (trajectory model)
    // double prob0_to_1 = sin_theta_half * sin_theta_half;
    // double prob1_to_1 = cos_theta_half * cos_theta_half;

    double prob0_to_1 = abs(sin_theta_half) / (abs(cos_theta_half) + abs(sin_theta_half));
    double prob1_to_1 = abs(cos_theta_half) / (abs(cos_theta_half) + abs(sin_theta_half));

    cuDoubleComplex exp_lambda     = make_cuDoubleComplex(cos(lambda),       sin(lambda));
    cuDoubleComplex exp_phi        = make_cuDoubleComplex(cos(phi),          sin(phi));
    cuDoubleComplex exp_phi_lambda = make_cuDoubleComplex(cos(lambda + phi), sin(lambda + phi));

    // Matrix entries
    cuDoubleComplex m00 = make_cuDoubleComplex(cos_theta_half, 0.0);
    cuDoubleComplex m01 = cuCmul(make_cuDoubleComplex(-sin_theta_half, 0.0), exp_lambda);
    cuDoubleComplex m10 = cuCmul(make_cuDoubleComplex( sin_theta_half, 0.0), exp_phi);
    cuDoubleComplex m11 = cuCmul(make_cuDoubleComplex( cos_theta_half, 0.0), exp_phi_lambda);
    double abs00 = cuCabs(m00);
    if (abs00 > 1e-12) m00 = make_cuDoubleComplex(m00.x / abs00, m00.y / abs00);

    double abs01 = cuCabs(m01);
    if (abs01 > 1e-12) m01 = make_cuDoubleComplex(m01.x / abs01, m01.y / abs01);

    double abs10 = cuCabs(m10);
    if (abs10 > 1e-12) m10 = make_cuDoubleComplex(m10.x / abs10, m10.y / abs10);

    double abs11 = cuCabs(m11);
    if (abs11 > 1e-12) m11 = make_cuDoubleComplex(m11.x / abs11, m11.y / abs11);


    int cur_state = get_qubit(row, qubit_index);
    double prob_to_be_1 = (cur_state == 1) ? prob1_to_1 : prob0_to_1;
    // double prob_to_be_1 = 0.5;
    

    float r = curand_uniform(&rand_states[tid]); // (0,1]
    int new_state = (r < prob_to_be_1) ? 1 : 0;

    set_qubit(row, qubit_index, new_state);

    // Weight update: pick amplitude corresponding to realized branch
    cuDoubleComplex new_weight =
        (cur_state == 0)
            ? ((new_state == 0) ? m00 : m10)
            : ((new_state == 0) ? m01 : m11);

    // --- DEBUGGING START ---
    // Only print for the first thread (or a specific one like tid == 10)
    // to avoid crashing the output buffer.
    // if (tid == 0) 
    // { 
    //     printf("TID %d\t| Qubit %d | cur_state: %d -> new_state: %d | Prob(1): %.4f | new_weight: %.4f+%.4fj | m00: %.4f + %.4fi | m01: %.4f + %.4fi | m10: %.4f + %.4fi | m11: %.4f + %.4fi\n", 
    //            tid, qubit_index, cur_state, new_state, prob_to_be_1, new_weight.x, new_weight.y, m00.x, m00.y, m01.x, m01.y, m10.x, m10.y, m11.x, m11.y);
    // }
    // --- DEBUGGING END ---

    weights[tid] = cuCmul(weights[tid], new_weight);
}

__global__ void cx_kernel(uint8_t* data, int num_shots, int num_bytes_per_shot,
                          int control_index, int target_index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* row = data + (size_t)tid * num_bytes_per_shot;

    int c = get_qubit(row, control_index);
    int t = get_qubit(row, target_index);
    set_qubit(row, target_index, c ^ t);
}

__global__ void cu3_kernel(uint8_t* data, curandState* rand_states,
                           int num_shots, int num_bytes_per_shot,
                           int control_index, int target_index,
                           double theta, double phi, double lambda) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    // If theta==0, no branching in this trajectory model (pure phase)
    if (theta == 0.0) return;

    uint8_t* row = data + (size_t)tid * num_bytes_per_shot;
    int c = get_qubit(row, control_index);
    if (c == 0) return;

    double cos_theta_half = cos(theta / 2.0);
    double sin_theta_half = sin(theta / 2.0);

    // double prob0_to_1 = sin_theta_half * sin_theta_half;
    // double prob1_to_1 = cos_theta_half * cos_theta_half;
    double prob0_to_1 = abs(sin_theta_half) / (abs(cos_theta_half) + abs(sin_theta_half));
    double prob1_to_1 = abs(cos_theta_half) / (abs(cos_theta_half) + abs(sin_theta_half));

    int t = get_qubit(row, target_index);
    double prob_to_be_1 = (t == 1) ? prob1_to_1 : prob0_to_1;

    float r = curand_uniform(&rand_states[tid]);
    int new_state = (r < prob_to_be_1) ? 1 : 0;
    set_qubit(row, target_index, new_state);
}

__global__ void pack_to_uint64_kernel(uint8_t* data, uint64_t* hashes,
                                      int num_shots, int num_bytes_per_shot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* row = data + (size_t)tid * num_bytes_per_shot;
    uint64_t h = 0;

    for (int i = 0; i < num_bytes_per_shot; ++i) {
        h |= (uint64_t)row[i] << (i * 8);
    }
    hashes[tid] = h;
}

// -------------------------
// prune+resample kernel
// -------------------------
__global__ void resample_from_kept_states_kernel(
    uint8_t* data,
    curandState* rand_states,
    cuDoubleComplex* weights,
    int num_shots,
    int num_bytes_per_shot,
    const uint64_t* kept_hashes,
    const double* kept_cdf,   // increasing, last == 1.0
    int num_kept
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    float u = curand_uniform(&rand_states[tid]); // (0,1]
    double x = (u == 1.0f) ? 0.999999999999 : (double)u;

    int idx = 0;
    while (idx < num_kept - 1 && x > kept_cdf[idx]) idx++;

    uint64_t h = kept_hashes[idx];

    uint8_t* row = data + (size_t)tid * num_bytes_per_shot;
    for (int i = 0; i < num_bytes_per_shot; ++i) {
        row[i] = (uint8_t)((h >> (i * 8)) & 0xFF);
    }

    weights[tid] = make_cuDoubleComplex(1.0, 0.0);
}

// ===================================================================
// == QASM PARSING UTILITIES (trim + angle parser)
// ===================================================================

static inline std::string& ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return s;
}
static inline std::string& rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}
static inline std::string& trim(std::string& s) {
    return ltrim(rtrim(s));
}

/**
 * @brief parse angles like: pi/2, -pi/4, 3*pi/4, 0.5
 */
static double parse_angle(std::string s) {
    trim(s);
    if (s.empty()) return 0.0;

    size_t pi_pos = s.find("pi");
    if (pi_pos != std::string::npos) {
        std::string before_pi = s.substr(0, pi_pos);
        std::string after_pi  = s.substr(pi_pos + 2);

        trim(before_pi);
        trim(after_pi);

        double multiplier = 1.0;
        double divisor    = 1.0;

        // multiplier
        if (!before_pi.empty()) {
            if (before_pi == "+") {
                multiplier = 1.0;
            } else if (before_pi == "-") {
                multiplier = -1.0;
            } else {
                if (!before_pi.empty() && before_pi.back() == '*') before_pi.pop_back();
                trim(before_pi);
                try { multiplier = std::stod(before_pi); }
                catch (...) { multiplier = 1.0; }
            }
        }

        // divisor like "/4"
        if (!after_pi.empty() && after_pi.rfind('/', 0) == 0) {
            std::string denom_str = after_pi.substr(1);
            trim(denom_str);
            try {
                divisor = std::stod(denom_str);
                if (divisor == 0.0) divisor = 1.0;
            } catch (...) {
                std::cerr << "Warning: Failed to parse pi denominator in '" << s
                          << "'. Defaulting to 1.\n";
                divisor = 1.0;
            }
        }

        return multiplier * M_PI / divisor;
    }

    // numeric only
    try { return std::stod(s); }
    catch (...) {
        std::cerr << "Warning: Could not parse angle '" << s << "'. Defaulting to 0.\n";
        return 0.0;
    }
}

// ===================================================================
// == HOST CLASS
// ===================================================================

class Trajectories {
public:
    // What we return for “measurement stats”
    struct MeasurementStat {
        std::string   ket;
        double        prob_raw;    // raw probability from this simulator (before any discard/renorm in main)
        cuDoubleComplex sum_w;     // summed weight for this basis (Σ weights over shots in this basis)
        cuDoubleComplex avg_w;     // average weight among shots in this basis (sum_w / count)
        int           count;       // number of shots whose final basis == ket
    };

    // --- Custom Thrust Functor for cuDoubleComplex ---
    struct cuDoubleComplex_add {
        __host__ __device__
        cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const {
            return cuCadd(a, b);
        }
    };

private:
    int num_qubits;
    int num_shots;
    int num_bytes_per_shot;

    std::map<std::string, int> qubit_register_offsets;

    uint8_t* d_data = nullptr;
    curandState* d_rand_states = nullptr;
    cuDoubleComplex* d_weights = nullptr;

    dim3 getGridDim() const { return dim3((num_shots + BLOCK_SIZE - 1) / BLOCK_SIZE); }
    dim3 getBlockDim() const { return dim3(BLOCK_SIZE); }

    std::string format_ket(uint64_t hash) const {
        std::string ket_str;
        ket_str.reserve(num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            int byte_idx = q / 8;
            int bit_idx  = q % 8;

            uint8_t byte = (hash >> (byte_idx * 8)) & 0xFF;
            int bit = (byte >> bit_idx) & 1;
            ket_str += (bit ? '1' : '0');
        }
        std::reverse(ket_str.begin(), ket_str.end());
        return "|" + ket_str + ">";
    }

    std::vector<int> parse_qubits(const std::string& args_str) const {
        std::vector<int> qubits;
        std::regex re("(\\w+)\\[(\\d+)\\]");

        std::sregex_iterator it(args_str.begin(), args_str.end(), re);
        std::sregex_iterator end;
        while (it != end) {
            std::smatch m = *it;
            std::string reg_name = m.str(1);
            int reg_index = std::stoi(m.str(2));

            auto map_it = qubit_register_offsets.find(reg_name);
            if (map_it == qubit_register_offsets.end()) {
                if (reg_name == "q" && qubit_register_offsets.size() == 1) {
                    qubits.push_back(qubit_register_offsets.begin()->second + reg_index);
                } else {
                    throw std::runtime_error("Error: unknown qreg '" + reg_name + "'");
                }
            } else {
                qubits.push_back(map_it->second + reg_index);
            }
            ++it;
        }
        return qubits;
    }

    // ------------------------------------------
    // prune + resample/reset all trajectories
    // Keep states whose probability >= threshold.
    // (still uses hash-prob only; no need for avg weight here)
    // ------------------------------------------
    void prune_and_resample_threshold(double prob_threshold) {
        auto probs = measure_state_probs_hash();
        if (probs.empty()) return;

        std::vector<std::pair<uint64_t,double>> kept;
        kept.reserve(probs.size());
        for (auto &p : probs) {
            if (p.second >= prob_threshold) kept.push_back(p);
        }

        if (kept.empty()) {
            auto best = *std::max_element(
                probs.begin(), probs.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; }
            );
            kept.push_back(best);
        }

        double sum = 0.0;
        for (auto &p : kept) sum += p.second;
        if (sum <= 0.0) {
            kept.clear();
            kept.push_back(probs[0]);
            sum = probs[0].second;
            if (sum <= 0.0) sum = 1.0;
        }

        std::vector<uint64_t> h_kept_hashes(kept.size());
        std::vector<double>   h_cdf(kept.size());

        double acc = 0.0;
        for (size_t i = 0; i < kept.size(); ++i) {
            h_kept_hashes[i] = kept[i].first;
            acc += kept[i].second / sum;
            h_cdf[i] = acc;
        }
        h_cdf.back() = 1.0;

        uint64_t* d_kept_hashes = nullptr;
        double*   d_kept_cdf = nullptr;

        checkCuda(cudaMalloc(&d_kept_hashes, h_kept_hashes.size() * sizeof(uint64_t)));
        checkCuda(cudaMalloc(&d_kept_cdf,    h_cdf.size()        * sizeof(double)));

        checkCuda(cudaMemcpy(d_kept_hashes, h_kept_hashes.data(),
                             h_kept_hashes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_kept_cdf, h_cdf.data(),
                             h_cdf.size() * sizeof(double), cudaMemcpyHostToDevice));

        resample_from_kept_states_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, d_rand_states, d_weights,
            num_shots, num_bytes_per_shot,
            d_kept_hashes, d_kept_cdf, (int)kept.size()
        );
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());

        checkCuda(cudaFree(d_kept_hashes));
        checkCuda(cudaFree(d_kept_cdf));
    }

    // ----------------------------
    // measure -> {hash, prob}   (used by pruning)
    // ----------------------------
    std::vector<std::pair<uint64_t, double>> measure_state_probs_hash() {
        checkCuda(cudaDeviceSynchronize());

        if (num_bytes_per_shot > 8) {
            throw std::runtime_error("measure_state_probs_hash() for > 64 qubits is not implemented.");
        }

        thrust::device_vector<uint64_t> d_hashes(num_shots);
        thrust::device_vector<uint64_t> d_unique_hashes(num_shots);
        thrust::device_vector<cuDoubleComplex> d_summed_weights(num_shots);

        pack_to_uint64_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, thrust::raw_pointer_cast(d_hashes.data()),
            num_shots, num_bytes_per_shot
        );
        checkCuda(cudaPeekAtLastError());

        thrust::device_ptr<cuDoubleComplex> d_weights_ptr(d_weights);
        thrust::sort_by_key(d_hashes.begin(), d_hashes.end(), d_weights_ptr);

        auto end = thrust::reduce_by_key(
            d_hashes.begin(), d_hashes.end(),
            d_weights_ptr,
            d_unique_hashes.begin(),
            d_summed_weights.begin(),
            thrust::equal_to<uint64_t>(),
            cuDoubleComplex_add()
        );

        int num_unique = (int)(end.first - d_unique_hashes.begin());

        std::vector<uint64_t> h_hashes(num_unique);
        std::vector<cuDoubleComplex> h_sumw(num_unique);

        thrust::copy(d_unique_hashes.begin(), d_unique_hashes.begin() + num_unique, h_hashes.begin());
        thrust::copy(d_summed_weights.begin(), d_summed_weights.begin() + num_unique, h_sumw.begin());

        double total = 0.0;
        std::vector<double> eff(num_unique);
        for (int i = 0; i < num_unique; ++i) {
            double a = cuCabs(h_sumw[i]);
            eff[i] = a * a;
            total += eff[i];
        }

        std::vector<std::pair<uint64_t, double>> out;
        out.reserve(num_unique);
        if (total <= 0.0) return out;

        for (int i = 0; i < num_unique; ++i) {
            out.push_back({h_hashes[i], eff[i] / total});
        }
        return out;
    }

    // --- QASM apply command returns "did execute a gate?" ---
    // --- QASM apply command returns "did execute a gate?" ---
    bool apply_qasm_command(const std::string& command,
                            const std::string& params,
                            const std::string& args) {
        std::vector<int> qubits = this->parse_qubits(args);

        if (command == "x" && qubits.size() == 1) {
            this->x(qubits[0]); return true;
        } else if (command == "h" && qubits.size() == 1) {
            this->h(qubits[0]); return true;
        } else if (command == "cx" && qubits.size() == 2) {
            this->cx(qubits[0], qubits[1]); return true;
        } else if (command == "sx" && qubits.size() == 1) {
            this->sx(qubits[0]); return true;
        } else if (command == "rz" && qubits.size() == 1) {
            double lambda = parse_angle(params);
            this->rz(qubits[0], lambda); return true;
        } else if (command == "t" && qubits.size() == 1) {
            this->t(qubits[0]); return true;
        } else if (command == "tdg" && qubits.size() == 1) {
            this->tdg(qubits[0]); return true;
        } else if (command == "s" && qubits.size() == 1) {
            this->s(qubits[0]); return true;
        } else if (command == "U" && qubits.size() == 1) {
            // NEW: Parse generic U gate "u(theta,phi,lambda)"
            // Params string comes in as "theta,phi,lambda" (commas, no parens)
            std::vector<std::string> tokens;
            std::stringstream ss(params);
            std::string token;
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            if (tokens.size() == 3) {
                double theta  = parse_angle(tokens[0]);
                double phi    = parse_angle(tokens[1]);
                double lambda = parse_angle(tokens[2]);
                this->u3(qubits[0], theta, phi, lambda);
                return true;
            } else {
                std::cerr << "Warning: 'u' gate expects 3 parameters (theta,phi,lambda), got " 
                          << tokens.size() << " in '" << params << "'\n";
                return false;
            }
        } else if (command == "barrier") {
            return false;
        } else if (command == "measure") {
            return false;
        }

        std::cerr << "Warning: Skipping unknown/unimplemented QASM command: '"
                  << command << " " << params << " " << args << "'\n";
        return false;
    }

public:
    static int get_qubits_and_map_from_qasm(const std::string& filename,
                                           std::map<std::string, int>& register_map) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open QASM file: " + filename);
        }

        register_map.clear();
        std::string line;
        std::regex re("qreg\\s+(\\w+)\\[(\\d+)\\]");
        int total_qubits = 0;
        bool found = false;

        while (std::getline(file, line)) {
            trim(line);
            std::smatch m;
            if (std::regex_search(line, m, re)) {
                std::string reg_name = m.str(1);
                int reg_size = std::stoi(m.str(2));
                if (register_map.count(reg_name)) {
                    throw std::runtime_error("Error: qreg redefined: " + reg_name);
                }
                register_map[reg_name] = total_qubits;
                total_qubits += reg_size;
                found = true;
            }
        }
        if (!found) {
            throw std::runtime_error("Error: no qreg found in: " + filename);
        }
        return total_qubits;
    }

    Trajectories(const std::string& qasm_filename, int n_shots = DEFAULT_TRAJECTORIES)
        : num_shots(n_shots) {

        std::cout << "Scanning " << qasm_filename << " for qubit count and registers...\n";
        this->num_qubits = get_qubits_and_map_from_qasm(qasm_filename, this->qubit_register_offsets);
        std::cout << "Found " << this->num_qubits << " qubits.\n";

        if (this->num_qubits <= 0) throw std::invalid_argument("num_qubits must be > 0");

        num_bytes_per_shot = (num_qubits + 7) / 8;

        size_t data_size   = (size_t)num_shots * num_bytes_per_shot * sizeof(uint8_t);
        size_t rand_size   = (size_t)num_shots * sizeof(curandState);
        size_t weight_size = (size_t)num_shots * sizeof(cuDoubleComplex);

        checkCuda(cudaMalloc(&d_data, data_size));
        checkCuda(cudaMalloc(&d_rand_states, rand_size));
        checkCuda(cudaMalloc(&d_weights, weight_size));

        checkCuda(cudaMemset(d_data, 0, data_size));

        unsigned long long seed = 1234ULL;
        init_curand_kernel<<<getGridDim(), getBlockDim()>>>(d_rand_states, num_shots, seed);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());

        init_weight_kernel<<<getGridDim(), getBlockDim()>>>(d_weights, num_shots);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());
    }

    ~Trajectories() {
        if (d_data) checkCuda(cudaFree(d_data));
        if (d_rand_states) checkCuda(cudaFree(d_rand_states));
        if (d_weights) checkCuda(cudaFree(d_weights));
    }

    int get_num_qubits() const { return num_qubits; }

    // --- Gates ---
    void x(int q) {
        x_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, q);
        checkCuda(cudaPeekAtLastError());
    }

    void u3(int q, double theta, double phi, double lambda) {

        u3_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, d_rand_states, d_weights,
            num_shots, num_bytes_per_shot,
            q, theta, phi, lambda);
            // q, theta, phi, lambda, global_phase
        checkCuda(cudaPeekAtLastError());
    }

    void h(int q) { this->u3(q, M_PI / 2.0, 0.0, M_PI); }

    void cx(int c, int t) {
        cx_kernel<<<getGridDim(), getBlockDim()>>>(d_data, num_shots, num_bytes_per_shot, c, t);
        checkCuda(cudaPeekAtLastError());
    }

    void cu3(int c, int t, double theta, double phi, double lambda) {
        cu3_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, d_rand_states, num_shots, num_bytes_per_shot,
            c, t, theta, phi, lambda
        );
        checkCuda(cudaPeekAtLastError());
    }

    void cp(int c, int t, double lambda) { this->cu3(c, t, 0.0, 0.0, lambda); }

    void t(int q)   { this->u3(q, 0.0, 0.0,  M_PI / 4.0); }
    void tdg(int q) { this->u3(q, 0.0, 0.0, -M_PI / 4.0); }
    void s(int q)   { this->u3(q, 0.0, 0.0,  M_PI / 2.0); }

    void sx(int q) {
        // double theta = M_PI/2.0;
        // double phi = -M_PI/2.0;
        // double lambda = M_PI/2.0;
        // double cos_theta_half = cos(theta / 2.0);
        // double sin_theta_half = sin(theta / 2.0);

        // // Probabilistic branching for basis state bit (trajectory model)
        // double prob0_to_1 = sin_theta_half * sin_theta_half;
        // double prob1_to_1 = cos_theta_half * cos_theta_half;

        // cuDoubleComplex exp_lambda     = make_cuDoubleComplex(cos(lambda),       sin(lambda));
        // cuDoubleComplex exp_phi        = make_cuDoubleComplex(cos(phi),          sin(phi));
        // cuDoubleComplex exp_phi_lambda = make_cuDoubleComplex(cos(lambda + phi), sin(lambda + phi));

        // // Matrix entries
        // cuDoubleComplex m00 = make_cuDoubleComplex(cos_theta_half, 0.0);
        // cuDoubleComplex m01 = cuCmul(make_cuDoubleComplex(-sin_theta_half, 0.0), exp_lambda);
        // cuDoubleComplex m10 = cuCmul(make_cuDoubleComplex( sin_theta_half, 0.0), exp_phi);
        // cuDoubleComplex m11 = cuCmul(make_cuDoubleComplex( cos_theta_half, 0.0), exp_phi_lambda);
        // print_cu("m00", m00);
        // print_cu("m01", m01);
        // print_cu("m10", m10);
        // print_cu("m11", m11);

        // cuDoubleComplex g = make_cuDoubleComplex(cos(M_PI/4.0), sin(M_PI/4.0));
        // cuDoubleComplex g = make_cuDoubleComplex(1, 0);
        this->u3(q, M_PI/2.0, -M_PI/2.0, M_PI/2.0);
    }

    void rz(int q, double lambda) {
        // cuDoubleComplex g = make_cuDoubleComplex(cos(-lambda/2.0), sin(-lambda/2.0));
        // cuDoubleComplex g = make_cuDoubleComplex(1, 0);

        this->u3(q, 0.0, 0.0, lambda);
    }
    static inline void print_cu(const char* name, cuDoubleComplex z) {
        std::cout << name << " = "
                << std::setprecision(17) << cuCreal(z)
                << (cuCimag(z) >= 0 ? " + " : " - ")
                << std::setprecision(17) << std::abs(cuCimag(z)) << "i"
                << "\n";
    }
    static inline void print_complex_host(const cuDoubleComplex& z) {
        std::cout << "(" << std::fixed << std::setprecision(12)
                << z.x << (z.y >= 0 ? "+" : "") << z.y << "i)";
    }

    // optional: show hash in hex to spot endian issues
    static inline void print_hash_hex(uint64_t h) {
        std::cout << "0x" << std::hex << h << std::dec;
    }

    struct HashWeightSort {
        __host__ __device__
        bool operator()(const thrust::tuple<uint64_t, cuDoubleComplex>& a,
                        const thrust::tuple<uint64_t, cuDoubleComplex>& b) const {
            uint64_t h_a = thrust::get<0>(a);
            uint64_t h_b = thrust::get<0>(b);
            
            // Primary sort key: Hash
            if (h_a != h_b) return h_a < h_b;

            // Secondary sort key: Weight (Real, then Imaginary)
            cuDoubleComplex w_a = thrust::get<1>(a);
            cuDoubleComplex w_b = thrust::get<1>(b);
            
            if (w_a.x != w_b.x) return w_a.x < w_b.x;
            return w_a.y < w_b.y;
        }
    };

    // Equality predicate for reducing (must match both Hash and Weight)
    struct HashWeightEqual {
        __host__ __device__
        bool operator()(const thrust::tuple<uint64_t, cuDoubleComplex>& a,
                        const thrust::tuple<uint64_t, cuDoubleComplex>& b) const {
            // Check Hash
            if (thrust::get<0>(a) != thrust::get<0>(b)) return false;

            // Check Weight (exact bitwise equality)
            cuDoubleComplex w_a = thrust::get<1>(a);
            cuDoubleComplex w_b = thrust::get<1>(b);
            return (w_a.x == w_b.x) && (w_a.y == w_b.y);
        }
    };
    // -------------------------------------------------------
    // NEW: measurement that returns BOTH probability and weight
    // - sum_w: reduced-by-key sum of weights for that ket
    // - count: number of shots in that ket
    // - avg_w: sum_w / count  (what you requested to print)
    // - prob_raw: |sum_w|^2 / Σ |sum_w|^2
    // -------------------------------------------------------
    // Functor to sum tuples of (cuDoubleComplex, int)
    // Used to reduce both Weights and Counts in one pass
    struct ComplexIntTupleAdd {
        __host__ __device__
        thrust::tuple<cuDoubleComplex, int> operator()(
            const thrust::tuple<cuDoubleComplex, int>& a,
            const thrust::tuple<cuDoubleComplex, int>& b
        ) const {
            cuDoubleComplex wa = thrust::get<0>(a);
            cuDoubleComplex wb = thrust::get<0>(b);
            int ca = thrust::get<1>(a);
            int cb = thrust::get<1>(b);
            
            // Complex Addition: (wa + wb)
            cuDoubleComplex w_sum = make_cuDoubleComplex(wa.x + wb.x, wa.y + wb.y);
            
            // Integer Addition: (ca + cb)
            return thrust::make_tuple(w_sum, ca + cb);
        }
    };

    std::vector<MeasurementStat> measure_stats() {
        checkCuda(cudaDeviceSynchronize());

        if (num_bytes_per_shot > 8) {
            throw std::runtime_error("measure_stats() for > 64 qubits is not implemented.");
        }

        // 1. Prepare raw storage
        thrust::device_vector<uint64_t> d_hashes(num_shots);
        
        // Output vectors for reduction
        thrust::device_vector<uint64_t> d_unique_hashes(num_shots);
        thrust::device_vector<cuDoubleComplex> d_summed_weights(num_shots);
        thrust::device_vector<int> d_counts(num_shots);

        // 2. Pack Hashes
        pack_to_uint64_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, thrust::raw_pointer_cast(d_hashes.data()),
            num_shots, num_bytes_per_shot
        );
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());

        // 3. Sort by Key (Hash Only)
        // We sort d_hashes, and move d_weights accordingly.
        thrust::device_ptr<cuDoubleComplex> d_weights_ptr(d_weights);
        thrust::sort_by_key(d_hashes.begin(), d_hashes.end(), d_weights_ptr);
        checkCuda(cudaDeviceSynchronize());

        // 4. Reduce by Key (Hash)
        // We want to:
        //  - Group by Hash
        //  - Sum the Weights
        //  - Sum the Counts (1 per shot)
        
        // Input Values: Zip(Weights, 1)
        auto begin_vals_in = thrust::make_zip_iterator(thrust::make_tuple(
            d_weights_ptr, 
            thrust::make_constant_iterator(1)
        ));
        
        // Output Values: Zip(SummedWeights, SummedCounts)
        auto begin_vals_out = thrust::make_zip_iterator(thrust::make_tuple(
            d_summed_weights.begin(), 
            d_counts.begin()
        ));

        auto end_pair = thrust::reduce_by_key(
            d_hashes.begin(), d_hashes.end(),       // Input Keys (Hash)
            begin_vals_in,                          // Input Values (Weight, 1)
            d_unique_hashes.begin(),                // Output Keys
            begin_vals_out,                         // Output Values
            thrust::equal_to<uint64_t>(),           // Key Equality Predicate
            ComplexIntTupleAdd()                    // Reduction Op (Sum w, Sum c)
        );

        int num_unique = (int)(end_pair.first - d_unique_hashes.begin());

        // 5. Copy back to Host
        std::vector<uint64_t> h_hashes(num_unique);
        std::vector<cuDoubleComplex> h_sum_w(num_unique);
        std::vector<int> h_cnt(num_unique);

        thrust::copy(d_unique_hashes.begin(), d_unique_hashes.begin() + num_unique, h_hashes.begin());
        thrust::copy(d_summed_weights.begin(), d_summed_weights.begin() + num_unique, h_sum_w.begin());
        thrust::copy(d_counts.begin(), d_counts.begin() + num_unique, h_cnt.begin());

        // 6. Debug Print
        std::cout << "\n[DEBUG] Aggregated Results (Grouped by Hash | Weight Summed | Total Groups: " << num_unique << "):\n";
        int n_print_reduced = std::min(num_unique, 16); 
        
        for (int i = 0; i < n_print_reduced; ++i) {
            std::cout << "  idx=" << std::setw(3) << i 
                    << " hash="; print_hash_hex(h_hashes[i]);
            std::cout << " ket=" << format_ket(h_hashes[i])
                    << " count=" << std::setw(5) << h_cnt[i]
                    << " sum_w="; print_complex_host(h_sum_w[i]);
            std::cout << "\n";
        }
        if (num_unique > n_print_reduced) {
            std::cout << "  ... (" << (num_unique - n_print_reduced) << " more groups omitted)\n";
        }

        // 7. Process Stats
        // Total shots (sum of counts)
        int total_shots = 0;
        cuDoubleComplex total_weight_all = make_cuDoubleComplex(0.0, 0.0);
        double total_mag = 0.0;
        for (int i = 0; i < num_unique; ++i) {
            total_shots += h_cnt[i];
            
            // Accumulate total weight (real and imag parts)
            total_weight_all.x += h_sum_w[i].x;
            total_weight_all.y += h_sum_w[i].y;
            double mag = cuCabs(h_sum_w[i]);
            total_mag += mag * mag;
        }
        total_mag = sqrt(total_mag);        

        std::cout << "total_shots = " << total_shots << std::endl;
        std::cout << "total_weight_all = "; print_complex_host(total_weight_all); std::cout << std::endl;
        std::cout << "total_mag = " << total_mag << std::endl;

        // Calculate magnitude of total weight if needed for normalization
        // double total_mag = cuCabs(total_weight_all); 

        std::vector<MeasurementStat> out;
        out.reserve(num_unique);
        if (total_shots <= 0) return out;

        for (int i = 0; i < num_unique; ++i) {
            MeasurementStat st;
            st.ket = format_ket(h_hashes[i]);
            st.sum_w = h_sum_w[i];
            st.count = h_cnt[i];
            
            // Average Weight (Contribution per shot)
            // Note: This divides the specific outcome's sum by the TOTAL shots
            st.avg_w = make_cuDoubleComplex(
                st.sum_w.x / total_mag, 
                st.sum_w.y / total_mag
            );

            // Probability Raw: Squared magnitude of the summed amplitude (Interference)
            // |sum_w|^2
            double mag = cuCabs(st.avg_w);
            st.prob_raw = mag * mag; 
            
            out.push_back(st);
        }
        return out;
    }

    // --- QASM runner with prune+resample every 100 executed gates ---
    void run_qasm_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open QASM file: " + filename);
        }

        const int PRUNE_INTERVAL = 100;
        // const double PRUNE_PROB_THRESHOLD = PRUNE_PROB_THRESHOLD; // keep states >= this during intermediate pruning
        int gate_count = 0;

        std::string line;
        while (std::getline(file, line)) {
            size_t comment_pos = line.find("//");
            if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);

            trim(line);

            if (line.empty() ||
                line.rfind("OPENQASM", 0) == 0 ||
                line.rfind("include", 0) == 0) {
                continue;
            }

            if (line.rfind("measure", 0) == 0) break;
            if (line.rfind("qreg", 0) == 0 || line.rfind("creg", 0) == 0) continue;

            if (!line.empty() && line.back() == ';') line.pop_back();

            std::string command, params, args;

            size_t param_open_pos  = line.find('(');
            size_t first_space_pos = line.find(' ');

            if (param_open_pos != std::string::npos &&
                (first_space_pos == std::string::npos || param_open_pos < first_space_pos)) {
                command = line.substr(0, param_open_pos);
                size_t param_close_pos = line.find(')');
                if (param_close_pos == std::string::npos) {
                    std::cerr << "Warning: malformed QASM (missing ')'): " << line << "\n";
                    continue;
                }
                params = line.substr(param_open_pos + 1, param_close_pos - param_open_pos - 1);

                size_t args_start_pos = line.find_first_not_of(' ', param_close_pos + 1);
                if (args_start_pos == std::string::npos) {
                    std::cerr << "Warning: malformed QASM (missing args): " << line << "\n";
                    continue;
                }
                args = line.substr(args_start_pos);
            } else if (first_space_pos != std::string::npos) {
                command = line.substr(0, first_space_pos);
                args    = line.substr(first_space_pos + 1);
            } else {
                command = line;
            }

            trim(command); trim(params); trim(args);

            bool did_gate = apply_qasm_command(command, params, args);
            if (did_gate) {
                gate_count++;
                if (gate_count % PRUNE_INTERVAL == 0) {
                    prune_and_resample_threshold(PRUNE_PROB_THRESHOLD);
                }
            }
        }

        file.close();
    }
};

// ===================================================================
// == MAIN
// ===================================================================

// static inline void print_complex(const cuDoubleComplex& z) {
//     std::cout << "(" << std::fixed << std::setprecision(12)
//               << z.x << (z.y >= 0 ? "+" : "") << z.y << "i)";
// }
// ---- helper: parse positive int shots ----
// static int parse_num_shots(const char* s) {
//     try {
//         std::string str(s);
//         trim(str);
//         if (str.empty()) throw std::invalid_argument("empty");
//         long long v = std::stoll(str);
//         if (v <= 0) throw std::out_of_range("shots must be > 0");
//         if (v > (long long)std::numeric_limits<int>::max())
//             throw std::out_of_range("shots too large for int");
//         return (int)v;
//     } catch (const std::exception& e) {
//         throw std::runtime_error(std::string("Invalid <num_shots>: '") + s +
//                                  "'. Must be a positive integer.");
//     }
// }

// ===================================================================
// == MAIN (Batch Processing)
// ===================================================================


// Helper: Parse log2 shots (e.g., input 10 -> 1024 shots)
// static int parse_shots_pow2(const char* s) {
//     try {
//         int p = std::stoi(s);
//         if (p < 0 || p > 30) throw std::runtime_error("Shots power must be 0-30");
//         return 1 << p;
//     } catch (...) {
//         throw std::runtime_error("Invalid shot power provided.");
//     }
// }
// Helper: Extract number from string for sorting (e.g. "qft_n5.qasm" -> 5)
int extract_n(const std::string& s) {
    std::smatch m;
    std::regex re("_n(\\d+)");
    if (std::regex_search(s, m, re)) return std::stoi(m.str(1));
    return 0;
}
// Helper to print complex numbers (copied from test function logic)
static inline void print_complex_res(const cuDoubleComplex& z) {
    std::cout << "(" << std::fixed << std::setprecision(12)
              << z.x << (z.y >= 0 ? "+" : "") << z.y << "i)";
}

int main(int argc, char* argv[]) {
    using clock = std::chrono::steady_clock;

    // Hardcoded directory per instructions
    std::string dir_path = "benchmarks";

    // Usage: 
    // ./qsim <prefix> <output_file> <0/1_show_results> [optional: log_shots]
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <prefix> <output_file> <0 or 1 (show results)> [optional: log_shots]\n";
        std::cerr << "Example (Auto shots): " << argv[0] << " qft results.txt 1\n";
        std::cerr << "Example (Fixed 2^10): " << argv[0] << " qft results.txt 0 10\n";
        return 1;
    }

    std::string prefix   = argv[1];
    std::string out_file = argv[2];
    
    // Parse Output Flag
    int show_results_flag = 0;
    try {
        show_results_flag = std::stoi(argv[3]);
        if (show_results_flag != 0 && show_results_flag != 1) {
            std::cerr << "Warning: Output flag should be 0 or 1. Defaulting to 0.\n";
            show_results_flag = 0;
        }
    } catch (...) {
        std::cerr << "Error: Invalid output flag.\n";
        return 1;
    }
    bool show_results = (show_results_flag == 1);

    // Check if user provided a specific shot count (log2)
    int fixed_log_shots = -1; // -1 indicates "use automatic mode"
    if (argc == 5) {
        try {
            fixed_log_shots = std::stoi(argv[4]);
            if (fixed_log_shots < 0) throw std::runtime_error("Negative shots");
        } catch (...) {
            std::cerr << "Error: Invalid log_shots argument. Must be a positive integer.\n";
            return 1;
        }
    }

    // 1. Collect and Sort Files
    if (!fs::exists(dir_path)) {
        std::cerr << "Error: Directory '" << dir_path << "' does not exist.\n";
        return 1;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".qasm") {
            std::string fname = entry.path().filename().string();
            if (fname.find(prefix) == 0) { 
                files.push_back(entry.path());
            }
        }
    }

    // Sort by qubit count
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        int na = extract_n(a.string());
        int nb = extract_n(b.string());
        if (na != nb) return na < nb;
        return a.string() < b.string();
    });

    if (files.empty()) {
        std::cout << "No files found with prefix '" << prefix << "' in " << dir_path << "\n";
        return 0;
    }

    // 2. Open Output File
    std::ofstream outfile(out_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file " << out_file << "\n";
        return 1;
    }
    
    outfile << "Filename,Qubits,TotalRuntime(s)\n";
    std::cout << "Found " << files.size() << " benchmarks in '" << dir_path << "'. Writing to " << out_file << "\n";
    if (fixed_log_shots != -1) {
        std::cout << "Mode: FIXED SHOTS (2^" << fixed_log_shots << ")\n";
    } else {
        std::cout << "Mode: DYNAMIC SHOTS (2^(n+5))\n";
    }
    std::cout << "Output Results: " << (show_results ? "YES" : "NO") << "\n\n";

    // 3. Loop and Bench
    for (const auto& fpath : files) {
        std::string filename = fpath.string();
        std::string shortname = fpath.filename().string();
        
        std::cout << ">>> Benchmarking: " << shortname << " ... \n";

        try {
            // A. Pre-scan to get qubit count
            std::map<std::string, int> temp_map;
            int num_qubits_detected = 0;
            try {
                num_qubits_detected = Trajectories::get_qubits_and_map_from_qasm(filename, temp_map);
            } catch (const std::exception& e) {
                std::cerr << "    [Error parsing QASM]: " << e.what() << "\n";
                outfile << shortname << ",ERROR_PARSE,0.0\n";
                continue;
            }

            // B. Determine Shot Count
            int NUM_SHOTS = 0;
            if (fixed_log_shots != -1) {
                NUM_SHOTS = 1 << fixed_log_shots;
            } else {
                int p = num_qubits_detected + 5;
                if (p > 30) p = 30; 
                NUM_SHOTS = 1 << p;
            }

            // C. Run Benchmark
            auto t_total0 = clock::now();

            // Scope for Trajectories object to ensure cleanup before next iteration
            {
                Trajectories circuit(filename, NUM_SHOTS);
                circuit.run_qasm_file(filename);
                auto stats = circuit.measure_stats();

                // --- OPTIONAL PRINTING LOGIC ---
                if (show_results) {
                    // const double FINAL_PROB_THRESHOLD = 1e-4; // adder
                    const double FINAL_PROB_THRESHOLD = PRUNE_PROB_THRESHOLD; // dnn, qft, ghz
                    std::vector<Trajectories::MeasurementStat> kept;
                    double kept_sum = 0.0;

                    for (const auto& st : stats) {
                        if (st.prob_raw >= FINAL_PROB_THRESHOLD) {
                            kept.push_back(st);
                            kept_sum += st.prob_raw;
                        }
                    }

                    // Fallback
                    if (kept.empty() && !stats.empty()) {
                        auto best_it = std::max_element(stats.begin(), stats.end(),
                            [](const auto& a, const auto& b){ return a.prob_raw < b.prob_raw; });
                        kept.push_back(*best_it);
                        kept_sum = best_it->prob_raw;
                    }

                    // Sort descending
                    std::sort(kept.begin(), kept.end(),
                        [](const auto& a, const auto& b){ return a.prob_raw > b.prob_raw; });

                    std::cout << "\n    --- Results (Top 10) ---\n";
                    for (size_t i = 0; i < kept.size() && i < 10; ++i) {
                        const auto& st = kept[i];
                        // double p_norm = (kept_sum > 0.0) ? (st.prob_raw / kept_sum) : 0.0;
                        double p_norm = st.prob_raw;

                        std::cout << "    " << std::setw(num_qubits_detected + 4) << std::left << st.ket
                                  << ": Prob = " << std::fixed << std::setprecision(6) << p_norm
                                  << "   avgW=";
                        print_complex_res(st.avg_w);
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
            }

            auto t_total1 = clock::now();
            double tot_sec = std::chrono::duration<double>(t_total1 - t_total0).count();

            std::cout << "    Done (" << std::fixed << std::setprecision(4) << tot_sec << " s)\n";

            outfile << shortname << "," << num_qubits_detected << "," << tot_sec << "\n";
            outfile.flush();

        } catch (const std::exception& e) {
            std::cerr << "    [FAILED]: " << e.what() << "\n";
            outfile << shortname << ",ERROR_RUN,0.0\n";
            outfile.flush();
        }
    }

    std::cout << "\nBatch Benchmark Complete.\n";
    outfile.close();
    return 0;
}