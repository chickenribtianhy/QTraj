#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept> // For exceptions
#include <algorithm> // For std::reverse
#include <cmath>       // For sin, cos, M_PI
#include <iomanip>     // For std::setw, std::fixed, std::setprecision

// --- NEW: Includes for QASM parsing ---
#include <fstream>   // For file I/O
#include <sstream>   // For string streams
#include <regex>     // For parsing qubit numbers

// --- CUDA Includes ---
#include <cuda_runtime.h>
#include <curand_kernel.h> // For on-GPU random numbers
#include <cuComplex.h>

// --- Thrust Includes (for measure()) ---
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_ptr.h>

// Define a default block size for kernels
#define BLOCK_SIZE 256
// Define a default number of trajectories if not specified
#define DEFAULT_TRAJECTORIES (1 << 10)

// --- CUDA Error Checking Macro ---
// A utility to wrap all CUDA API calls for robust error checking
#define checkCuda(result) { \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s - %s:%d\n", cudaGetErrorString(result), __FILE__, __LINE__); \
        exit(result); \
    } \
}

// ===================================================================
// == CUDA KERNELS & DEVICE FUNCTIONS
// ===================================================================

/**
 * @brief (Device Function) Gets the state of a single qubit from a shot's data.
 * This is a helper function to be called from within a kernel.
 * * @param shot_row Pointer to the start of the data for a single trajectory.
 * @param qubit_index The qubit to read (0 to num_qubits-1).
 * @return int The qubit's state (0 or 1).
 */
__device__ inline int get_qubit(const uint8_t* shot_row, int qubit_index) {
    int byte_index = qubit_index / 8;
    int bit_index = qubit_index % 8;
    uint8_t mask = 1 << bit_index;
    return (shot_row[byte_index] & mask) >> bit_index;
}

/**
 * @brief (Device Function) Sets the state of a single qubit in a shot's data.
 * This is a helper function to be called from within a kernel.
 * * @param shot_row Pointer to the start of the data for a single trajectory.
 * @param qubit_index The qubit to write to (0 to num_qubits-1).
 * @param new_state The new state to write (0 or 1).
 */
__device__ inline void set_qubit(uint8_t* shot_row, int qubit_index, int new_state) {
    int byte_index = qubit_index / 8;
    int bit_index = qubit_index % 8;
    uint8_t mask = 1 << bit_index;
    
    // Read the current byte, clear the bit, and set it to the new state
    // Note: This is NOT atomic, but each thread works on a different shot_row,
    // so no race condition exists between threads.
    uint8_t byte = shot_row[byte_index];
    byte = (byte & ~mask) | (new_state << bit_index);
    shot_row[byte_index] = byte;
}

/**
 * @brief (Kernel) Initializes the cuRAND states for each trajectory.
 * One state is created per thread (per trajectory).
 */
__global__ void init_curand_kernel(curandState* states, int num_states, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_states) {
        // Initialize the state with a unique seed per thread
        curand_init(seed, tid, 0, &states[tid]);
    }
}

__global__ void init_weight_kernel(cuDoubleComplex* weights, int num_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_states) {
        weights[tid] = make_cuDoubleComplex(1, 0);
    }
}

/**
 * @brief (Kernel) Applies the X (NOT) gate to a qubit.
 */
__global__ void x_kernel(uint8_t* data, int num_shots, int num_bytes_per_shot, int qubit_index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* my_shot_row = data + (size_t)tid * num_bytes_per_shot;
    int current_state = get_qubit(my_shot_row, qubit_index);
    set_qubit(my_shot_row, qubit_index, 1 - current_state);
}

/**
 * @brief (Kernel) Applies the U3 gate to a qubit.
 */
__global__ void u3_kernel_old(uint8_t* data, curandState* rand_states, int num_shots, int num_bytes_per_shot, 
                          int qubit_index, double theta, double phi, double lambda) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    // Per your Python logic, only `theta` causes branching.
    // If theta is 0, this kernel effectively does nothing, which is correct
    // for phase gates (t, s, tdg, cp) in this probabilistic model.
    if (theta == 0.0) {
        return;
    }

    uint8_t* my_shot_row = data + (size_t)tid * num_bytes_per_shot;
    
    // Calculate probabilities
    double cos_theta_half = cos(theta / 2.0);
    double sin_theta_half = sin(theta / 2.0);
    double prob0_to_1 = sin_theta_half * sin_theta_half;
    double prob1_to_1 = cos_theta_half * cos_theta_half;

    int current_state = get_qubit(my_shot_row, qubit_index);
    double prob_to_be_1 = (current_state == 1) ? prob1_to_1 : prob0_to_1;

    // Get a random number for this thread
    float rand_val = curand_uniform(&rand_states[tid]);

    // Branch
    int new_state = (rand_val < prob_to_be_1) ? 1 : 0;
    
    set_qubit(my_shot_row, qubit_index, new_state);
}

__global__ void u3_kernel(uint8_t* data, curandState* rand_states, cuDoubleComplex* weights, int num_shots, int num_bytes_per_shot, 
                          int qubit_index, double theta, double phi, double lambda) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* my_shot_row = data + (size_t)tid * num_bytes_per_shot;

    // Apply the U3 matrix transformation:
    // | new_alpha | = | cos(t/2)            -e^(i*l)sin(t/2)    | | alpha_old |
    // | new_beta  |   | e^(i*p)sin(t/2)     e^(i*(p+l))cos(t/2) | | beta_old  |
    
    double cos_theta_half = cos(theta / 2.0);
    double sin_theta_half = sin(theta / 2.0);

    double prob0_to_1 = sin_theta_half * sin_theta_half;
    double prob1_to_1 = cos_theta_half * cos_theta_half;

    cuDoubleComplex exp_lambda = make_cuDoubleComplex(cos(lambda), sin(lambda));
    cuDoubleComplex exp_phi = make_cuDoubleComplex(cos(phi), sin(phi));
    cuDoubleComplex exp_phi_lambda = make_cuDoubleComplex(cos(lambda+phi), sin(lambda+phi));

    // m00 = cos(t/2)
    cuDoubleComplex m00 = make_cuDoubleComplex(cos_theta_half, 0.0);
    // m01 = -e^(i*l) * sin(t/2)
    cuDoubleComplex m01 = cuCmul(make_cuDoubleComplex(-sin_theta_half, 0.0), exp_lambda);
    // m10 = e^(i*p) * sin(t/2)
    cuDoubleComplex m10 = cuCmul(make_cuDoubleComplex(sin_theta_half, 0.0), exp_phi);
    // m11 = e^(i*(p+l)) * cos(t/2)
    cuDoubleComplex m11 = cuCmul(make_cuDoubleComplex(cos_theta_half, 0.0), exp_phi_lambda);
    

    int current_state = get_qubit(my_shot_row, qubit_index);
    double prob_to_be_1 = (current_state == 1) ? prob1_to_1 : prob0_to_1;

    // Get a random number for this thread
    float rand_val = curand_uniform(&rand_states[tid]);

    // Branch
    int new_state = (rand_val < prob_to_be_1) ? 1 : 0;
    
    set_qubit(my_shot_row, qubit_index, new_state);

    cuDoubleComplex new_weight = (current_state == 0) ? ((new_state == 0) ? m00 : m10) : ((new_state == 0) ? m01 : m11);
    weights[tid] = cuCmul(weights[tid], new_weight);
}

/**
 * @brief (Kernel) Applies the CX (CNOT) gate.
 */
__global__ void cx_kernel(uint8_t* data, int num_shots, int num_bytes_per_shot, int control_index, int target_index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* my_shot_row = data + (size_t)tid * num_bytes_per_shot;
    
    int control_state = get_qubit(my_shot_row, control_index);
    int target_state = get_qubit(my_shot_row, target_index);
    
    // XOR logic for CNOT
    set_qubit(my_shot_row, target_index, control_state ^ target_state);
}

/**
 * @brief (Kernel) Applies the controlled-U3 gate.
 */
__global__ void cu3_kernel(uint8_t* data, curandState* rand_states, int num_shots, int num_bytes_per_shot, 
                           int control_index, int target_index, double theta, double phi, double lambda) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    // Per your Python logic, only `theta` causes branching.
    if (theta == 0.0) {
        return;
    }

    uint8_t* my_shot_row = data + (size_t)tid * num_bytes_per_shot;

    int control_state = get_qubit(my_shot_row, control_index);

    // Only apply the U3 logic if the control qubit is 1
    if (control_state == 1) {
        // Calculate probabilities
        double cos_theta_half = cos(theta / 2.0);
        double sin_theta_half = sin(theta / 2.0);
        double prob0_to_1 = sin_theta_half * sin_theta_half;
        double prob1_to_1 = cos_theta_half * cos_theta_half;

        int target_state = get_qubit(my_shot_row, target_index);
        double prob_to_be_1 = (target_state == 1) ? prob1_to_1 : prob0_to_1;

        // Get a random number for this thread
        float rand_val = curand_uniform(&rand_states[tid]);

        // Branch
        int new_state = (rand_val < prob_to_be_1) ? 1 : 0;
        
        set_qubit(my_shot_row, target_index, new_state);
    }
    // If control_state is 0, do nothing.
}

/**
 * @brief (Kernel) Packs the byte-array states into single uint64_t integers for hashing.
 * This is the first step of the `measure` operation.
 */
__global__ void pack_to_uint64_kernel(uint8_t* data, uint64_t* hashes, int num_shots, int num_bytes_per_shot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shots) return;

    uint8_t* my_shot_row = data + (size_t)tid * num_bytes_per_shot;
    uint64_t hash = 0;

    // This loop packs the bytes into a uint64_t.
    // my_shot_row[0] (which contains qubits 0-7) becomes the LEAST significant byte.
    // This matches the simplified hashing logic.
    for (int i = 0; i < num_bytes_per_shot; ++i) {
        hash |= (uint64_t)my_shot_row[i] << (i * 8);
    }

    hashes[tid] = hash;
}


// ===================================================================
// == QASM PARSING UTILITIES
// ===================================================================

// --- String trimming functions ---
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
 * @brief Helper to parse simple math expressions from QASM like "pi/4" or "0.5"
 */
static double parse_angle(std::string s) {
    trim(s);
    if (s.empty()) return 0.0;

    // Simple case: just a number
    if (s.find("pi") == std::string::npos) {
        try { return std::stod(s); }
        catch (...) { return 0.0; }
    }

    // Handle "pi" expressions
    if (s == "pi") return M_PI;
    if (s == "-pi") return -M_PI;
    
    // Handle "pi/X"
    if (s.rfind("pi/", 0) == 0) {
        std::string denom = s.substr(3);
        try { return M_PI / std::stod(denom); }
        catch (...) { return 0.0; }
    }
    // Handle "-pi/X"
    if (s.rfind("-pi/", 0) == 0) {
        std::string denom = s.substr(4);
        try { return -M_PI / std::stod(denom); }
        catch (...) { return 0.0; }
    }
    
    // Handle "X*pi"
    size_t star_pi_pos = s.find("*pi");
    if (star_pi_pos != std::string::npos) {
        std::string mult = s.substr(0, star_pi_pos);
        try { return std::stod(mult) * M_PI; }
        catch (...) { return 0.0; }
    }

    std::cerr << "Warning: Could not fully parse angle '" << s << "'. Defaulting to 0." << std::endl;
    return 0.0;
}

// ===================================================================
// == HOST CLASS
// ===================================================================

class Trajectories {
private:
    int num_qubits;
    int num_shots;
    int num_bytes_per_shot; // (num_qubits + 7) / 8

    // NEW: Qubit register mapping
    std::map<std::string, int> qubit_register_offsets;

    // --- Device (GPU) Data Pointers ---
    uint8_t* d_data;            // Main state data: [num_shots * num_bytes_per_shot]
    curandState* d_rand_states; // cuRAND states: [num_shots]
    cuDoubleComplex* d_weights;       // accumulative weights: [num_shots]

    // --- Helper to calculate kernel launch parameters ---
    dim3 getGridDim() const {
        return dim3((num_shots + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }

    dim3 getBlockDim() const {
        return dim3(BLOCK_SIZE);
    }

    /**
     * @brief Converts a 64-bit hash back into a bitstring |q(n-1)...q1q0>.
     */
    std::string format_ket(uint64_t hash) const {
        std::string ket_str;
        ket_str.reserve(num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            int byte_idx = q / 8;
            int bit_idx = q % 8;
            
            // Get the byte, then the bit
            uint8_t byte = (hash >> (byte_idx * 8)) & 0xFF;
            int bit = (byte >> bit_idx) & 1;
            
            ket_str += (bit ? '1' : '0');
        }
        // Reverse to match the |q(n-1)...q0> convention
        std::reverse(ket_str.begin(), ket_str.end());
        return "|" + ket_str + ">";
    }

    /**
     * @brief (MEMBER FUNCTION) Helper to parse qubit indices from strings like "a[2]" or "b[1],cout[0]"
     */
    std::vector<int> parse_qubits(const std::string& args_str) const {
        std::vector<int> qubits;
        // Regex to find "NAME[INDEX]"
        std::regex re("(\\w+)\\[(\\d+)\\]"); 
        
        std::sregex_iterator next(args_str.begin(), args_str.end(), re);
        std::sregex_iterator end;
        while (next != end) {
            std::smatch match = *next;
            std::string reg_name = match.str(1);
            int reg_index = std::stoi(match.str(2));
            
            // Find the register's base offset
            auto it = qubit_register_offsets.find(reg_name);
            if (it == qubit_register_offsets.end()) {
                // Special case: QASM files might use 'q' for a generic register
                // that was defined with another name. This is brittle, but
                // we'll check for 'q' if the name isn't found, assuming it's
                // the *only* register.
                if (reg_name == "q" && qubit_register_offsets.size() == 1) {
                    qubits.push_back(qubit_register_offsets.begin()->second + reg_index);
                } else {
                    throw std::runtime_error("Error: QASM parsing found unknown qubit register '" + reg_name + "'");
                }
            } else {
                int base_offset = it->second;
                qubits.push_back(base_offset + reg_index); // Add the final, flat index
            }
            
            next++;
        }
        return qubits;
    }


    /**
     * @brief NEW: Internal method to apply a parsed QASM command.
     */
    void apply_qasm_command(const std::string& command, const std::string& params, const std::string& args) {
        // std::cout << "Applying command: " << command << " with params: " << params << " and args: " << args << std::endl;
        
        // NOW CALLS THE MEMBER FUNCTION
        std::vector<int> qubits = this->parse_qubits(args);

        if (command == "x" && qubits.size() == 1) {
            this->x(qubits[0]);
        }
        else if (command == "h" && qubits.size() == 1) {
            this->h(qubits[0]);
        }
        else if (command == "cx" && qubits.size() == 2) {
            this->cx(qubits[0], qubits[1]);
        }
        // --- NEW GATES ---
        else if (command == "sx" && qubits.size() == 1) {
            this->sx(qubits[0]);
        }
        else if (command == "rz" && qubits.size() == 1) {
            double lambda = parse_angle(params);
            this->rz(qubits[0], lambda);
        }
        // --- END NEW GATES ---
        else if (command == "t" && qubits.size() == 1) {
            this->t(qubits[0]);
        }
        else if (command == "tdg" && qubits.size() == 1) {
            this->tdg(qubits[0]);
        }
        else if (command == "s" && qubits.size() == 1) {
            this->s(qubits[0]);
        }
        else if (command == "barrier") {
            // Barrier is a directive, not a gate.
            // It has no effect in this simulator.
            // We recognize it here to avoid the "unknown command" warning.
            // pass
        }
        // --- Other gates (add skeletons as needed) ---
        else if (command == "measure") {
            // Measure is handled at the end by the .measure() call
            // In a real simulator, this would map qreg to creg
            // For this model, we can just ignore it.
        }
        else {
            std::cerr << "Warning: Skipping unknown or unimplemented QASM command: '" 
                      << command << " " << params << " " << args << "'" << std::endl;
        }
    }


public:
    /**
     * @brief NEW: Static utility to pre-scan a QASM file for qubit count and register map.
     */
    static int get_qubits_and_map_from_qasm(const std::string& filename, std::map<std::string, int>& register_map) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open QASM file to read qubit count: " + filename);
        }

        register_map.clear();
        std::string line;
        std::regex re("qreg\\s+(\\w+)\\[(\\d+)\\]"); // Regex for "qreg NAME[SIZE]"
        int total_qubits = 0;
        bool found_qreg = false;

        while (std::getline(file, line)) {
            trim(line);
            std::smatch match;
            if (std::regex_search(line, match, re)) {
                if (match.size() == 3) {
                    std::string reg_name = match.str(1);
                    int reg_size = std::stoi(match.str(2));
                    
                    if (register_map.count(reg_name)) {
                         throw std::runtime_error("Error: QASM file re-defines qreg '" + reg_name + "'");
                    }
                    
                    // Store the base offset for this register
                    register_map[reg_name] = total_qubits; 
                    total_qubits += reg_size; // Add the size
                    found_qreg = true;
                }
            }
        }
        
        if (!found_qreg) {
            throw std::runtime_error("Error: Could not find any 'qreg' definitions in QASM file: " + filename);
        }
        
        return total_qubits;
    }

    /**
     * @brief NEW: Constructor that takes a QASM file to configure itself.
     */
    Trajectories(const std::string& qasm_filename, int n_shots = DEFAULT_TRAJECTORIES)
        : num_shots(n_shots) {

        // 1. Pre-scan file to get qubit count and register map
        std::cout << "Scanning " << qasm_filename << " for qubit count and registers..." << std::endl;
        this->num_qubits = get_qubits_and_map_from_qasm(qasm_filename, this->qubit_register_offsets);
        std::cout << "Found " << this->num_qubits << " qubits." << std::endl;
        
        // Debug: Print the map
        // std::cout << "Register map:" << std::endl;
        // for(const auto& pair : this->qubit_register_offsets) {
        //     std::cout << "  " << pair.first << " -> offset " << pair.second << std::endl;
        // }


        // 2. Continue with original constructor logic
        if (this->num_qubits <= 0) {
            throw std::invalid_argument("Number of qubits must be a positive integer.");
        }
        
        num_bytes_per_shot = (num_qubits + 7) / 8;
        size_t data_size = (size_t)num_shots * num_bytes_per_shot * sizeof(uint8_t);
        size_t rand_states_size = (size_t)num_shots * sizeof(curandState);
        size_t weight_size = (size_t)num_shots * sizeof(cuDoubleComplex);

        // 3. Allocate device memory
        checkCuda( cudaMalloc(&d_data, data_size) );
        checkCuda( cudaMalloc(&d_rand_states, rand_states_size) );
        checkCuda( cudaMalloc(&d_weights, weight_size) );

        // 4. Initialize data to all-zero (all |0> states)
        checkCuda( cudaMemset(d_data, 0, data_size) );

        // 5. Initialize cuRAND states
        unsigned long long seed = 1234ULL; // You can use time(NULL) for a different seed
        init_curand_kernel<<<getGridDim(), getBlockDim()>>>(d_rand_states, num_shots, seed);
        checkCuda( cudaPeekAtLastError() );
        checkCuda( cudaDeviceSynchronize() ); // Wait for init to finish

        // 6. Initialize weights to 1
        init_weight_kernel<<<getGridDim(), getBlockDim()>>>(d_weights, num_shots);
        checkCuda( cudaPeekAtLastError() );
        checkCuda( cudaDeviceSynchronize() ); // Wait for init to finish
    }

    ~Trajectories() {
        // Destructor automatically handles freeing GPU memory
        // std::cout << "Trajectories object deconstructed. Releasing memory." << std::endl;
        checkCuda( cudaFree(d_data) );
        checkCuda( cudaFree(d_rand_states) );
        checkCuda( cudaFree(d_weights) );
    }

    // --- Public Getters ---
    int get_num_qubits() const { return num_qubits; }

    // --- Public Gate Methods ---

    void x(int qubit_index) {
        x_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, num_shots, num_bytes_per_shot, qubit_index
        );
        checkCuda( cudaPeekAtLastError() );
    }

    void h(int qubit_index) {
        // H = U3(pi/2, 0, pi)
        this->u3(qubit_index, M_PI / 2.0, 0.0, M_PI);
    }

    void u3(int qubit_index, double theta, double phi, double lambda) {
        u3_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, d_rand_states, d_weights, num_shots, num_bytes_per_shot, 
            qubit_index, theta, phi, lambda
        );
        checkCuda( cudaPeekAtLastError() );
    }

    void cx(int control_index, int target_index) {
        cx_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, num_shots, num_bytes_per_shot, control_index, target_index
        );
        checkCuda( cudaPeekAtLastError() );
    }

    void cu3(int control_index, int target_index, double theta, double phi, double lambda) {
        cu3_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, d_rand_states, num_shots, num_bytes_per_shot,
            control_index, target_index, theta, phi, lambda
        );
        checkCuda( cudaPeekAtLastError() );
    }

    void cp(int control_index, int target_index, double lambda) {
        // CP = CU3(0, 0, lambda)
        this->cu3(control_index, target_index, 0.0, 0.0, lambda);
    }

    // --- NEW GATES (SKELETONS) ---
    // These are phase gates, which are U3 gates with theta=0.
    // In this simulator, they correctly do nothing to the probabilities.

    /**
     * @brief T-gate (pi/8 phase rotation)
     * QASM: t q[i]
     */
    void t(int qubit_index) {
        // T = U3(0, 0, pi/4)
        this->u3(qubit_index, 0.0, 0.0, M_PI / 4.0);
    }

    /**
     * @brief T-dagger-gate (inverse T-gate)
     * QASM: tdg q[i]
     */
    void tdg(int qubit_index) {
        // Tdg = U3(0, 0, -pi/4)
        this->u3(qubit_index, 0.0, 0.0, -M_PI / 4.0);
    }

    /**
 * @brief S-gate (pi/2 phase rotation)
     * QASM: s q[i]
     */
    void s(int qubit_index) {
        // S = U3(0, 0, pi/2)
        this->u3(qubit_index, 0.0, 0.0, M_PI / 2.0);
    }

    /**
     * @brief SX-gate (sqrt-X)
     * QASM: sx q[i]
     */
    void sx(int qubit_index) {
        // SX = U3(pi/2, -pi/2, pi/2)
        this->u3(qubit_index, M_PI / 2.0, -M_PI / 2.0, M_PI / 2.0);
    }

    /**
     * @brief Rz-gate (Z-axis rotation)
     * QASM: rz(lambda) q[i]
     */
    void rz(int qubit_index, double lambda) {
        // Rz = U3(0, 0, lambda) (up to global phase)
        // In this probabilistic simulator, theta=0 means no change
        // in the |0>/|1> probabilities, which is correct.
        this->u3(qubit_index, 0.0, 0.0, lambda);
    }


    // --- Measurement ---

    std::map<std::string, int> measure_old() {
        // Synchronize device to make sure all kernels are finished before measuring
        checkCuda( cudaDeviceSynchronize() );

        // Fast Path: <= 64 qubits (8 bytes)
        // We can pack each shot's state into a single uint64_t and use Thrust
        // to sort-and-reduce, which is extremely fast.
        if (num_bytes_per_shot > 8) {
            throw std::runtime_error("Measure() for > 64 qubits is not implemented. (Matches Python 'slow path' warning)");
        }

        // 1. Create Thrust device vectors for the hash-and-reduce operation
        thrust::device_vector<uint64_t> d_hashes(num_shots);
        thrust::device_vector<uint64_t> d_unique_hashes(num_shots);
        thrust::device_vector<int> d_counts(num_shots);

        // 2. Launch kernel to pack all byte rows into uint64_t hashes
        pack_to_uint64_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, thrust::raw_pointer_cast(d_hashes.data()), 
            num_shots, num_bytes_per_shot
        );
        checkCuda( cudaPeekAtLastError() );

        // 3. Sort the hashes
        // This brings all identical states next to each other
        thrust::sort(d_hashes.begin(), d_hashes.end());

        // 4. Reduce by key (the "unique" and "count" operation)
        // This finds the unique hashes and counts their occurrences
        auto end = thrust::reduce_by_key(
            d_hashes.begin(), d_hashes.end(),           // Input keys (the sorted hashes)
            thrust::make_constant_iterator(1),         // Input values (just "1" for each hash)
            d_unique_hashes.begin(),                   // Output keys (the unique hashes)
            d_counts.begin()                           // Output values (the counts)
        );

        // `end.first` points to the end of the unique hashes
        int num_unique_states = end.first - d_unique_hashes.begin();
        
        // 5. Copy results from GPU (Thrust vectors) to CPU (std::vectors)
        std::vector<uint64_t> h_unique_hashes(num_unique_states);
        std::vector<int> h_counts(num_unique_states);

        thrust::copy(d_unique_hashes.begin(), d_unique_hashes.begin() + num_unique_states, h_unique_hashes.begin());
        thrust::copy(d_counts.begin(), d_counts.begin() + num_unique_states, h_counts.begin());

        // 6. Format results into the final map on the CPU
        std::map<std::string, int> results;
        for (int i = 0; i < num_unique_states; ++i) {
            std::string ket = format_ket(h_unique_hashes[i]);
            results[ket] = h_counts[i];
        }

        return results;
    }
    
    // --- Custom Thrust Functor for cuDoubleComplex ---
    // This struct tells Thrust how to add two cuDoubleComplex values,
    // as the default 'thrust::plus' (which uses '+') fails to compile.
    struct cuDoubleComplex_add
    {
        __host__ __device__
        cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const
        {
            return cuCadd(a, b);
        }
    };
    // MODIFIED: Returns a map of <state, probability>
    std::map<std::string, double> measure() {
        // Synchronize device to make sure all kernels are finished before measuring
        checkCuda( cudaDeviceSynchronize() );

        if (num_bytes_per_shot > 8) {
            throw std::runtime_error("Measure() for > 64 qubits is not implemented.");
        }

        // 1. Create Thrust device vectors for intermediate/output storage
        // We still need a place to store the hashes.
        thrust::device_vector<uint64_t> d_hashes(num_shots);
        
        // These will store the output of the reduction
        thrust::device_vector<uint64_t> d_unique_hashes(num_shots);
        thrust::device_vector<cuDoubleComplex> d_summed_weights(num_shots);

        // 2. Launch kernel to pack all byte rows into uint64_t hashes
        pack_to_uint64_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, thrust::raw_pointer_cast(d_hashes.data()), 
            num_shots, num_bytes_per_shot
        );
        checkCuda( cudaPeekAtLastError() );

        // 3. NEW: Wrap the existing raw device pointers for Thrust
        // d_hashes is already a device_vector, so d_hashes.begin() works
        // d_weights is a raw pointer, so we wrap it.
        thrust::device_ptr<cuDoubleComplex> d_weights_ptr(d_weights);

        // 3. Sort by key (the hashes)
        // This sorts d_hashes (in place)
        // and reorders the data pointed to by d_weights_ptr (also in place)
        thrust::sort_by_key(
            d_hashes.begin(), d_hashes.end(),   // Keys to sort by
            d_weights_ptr                       // Values to sort in parallel
        );

        // 4. Reduce by key (the "unique" and "sum" operation)
        // This finds the unique hashes and SUMS their corresponding weights
        auto end = thrust::reduce_by_key(
            d_hashes.begin(), d_hashes.end(),           // Input keys (sorted hashes)
            d_weights_ptr,                             // Input values (sorted weights)
            d_unique_hashes.begin(),                   // Output keys (unique hashes)
            d_summed_weights.begin(),                   // Output values (SUMMED weights)
            thrust::equal_to<uint64_t>(),              // <-- (NEW) How to compare keys
            cuDoubleComplex_add()                      // <-- (NEW) How to sum values
        );

        // `end.first` points to the end of the unique hashes
        int num_unique_states = end.first - d_unique_hashes.begin();
        
        // 5. Copy results from GPU (Thrust vectors) to CPU (std::vectors)
        std::vector<uint64_t> h_unique_hashes(num_unique_states);
        std::vector<cuDoubleComplex> h_summed_weights(num_unique_states);

        thrust::copy(d_unique_hashes.begin(), d_unique_hashes.begin() + num_unique_states, h_unique_hashes.begin());
        thrust::copy(d_summed_weights.begin(), d_summed_weights.begin() + num_unique_states, h_summed_weights.begin());

        // 6. Format results into the final map on the CPU
        std::map<std::string, double> results;
        double total_shots_double = static_cast<double>(num_shots);
        cuDoubleComplex total_shots_complex = make_cuDoubleComplex(total_shots_double, 0.0);

        for (int i = 0; i < num_unique_states; ++i) {
            std::string ket = format_ket(h_unique_hashes[i]);
            
            // W_k_total = Sum(weights) for state k
            cuDoubleComplex W_k_total = h_summed_weights[i];
            
            // W_k_avg = W_k_total / N_shots
            cuDoubleComplex W_k_avg = cuCdiv(W_k_total, total_shots_complex);
            
            // prob_k = |W_k_avg|^2
            double prob_k = cuCabs(W_k_avg) * cuCabs(W_k_avg);
            
            results[ket] = prob_k;
        }

        return results;
    }

    // --- NEW: QASM FILE EXECUTION ---

    /**
     * @brief Reads a QASM file and applies the operations.
     */
    void run_qasm_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open QASM file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            // Remove comments
            size_t comment_pos = line.find("//");
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            
            // Trim whitespace
            trim(line);

            // Skip empty lines or headers
            if (line.empty() || 
                line.rfind("OPENQASM", 0) == 0 || 
                line.rfind("include", 0) == 0) {
                continue;
            }

            // Stop at measurement; we do that separately
            if (line.rfind("measure", 0) == 0) {
                // In this simulator, all measurements happen at the end
                break;
            }

            // Skip register definitions (we pre-parsed qreg in the constructor)
            if (line.rfind("qreg", 0) == 0 || line.rfind("creg", 0) == 0) {
                continue;
            }

            // Parse the command, parameters, and arguments
            std::string command;
            std::string params;
            std::string args;
            
            trim(line); // Ensure line is trimmed
            
            // Remove trailing semicolon
            if (!line.empty() && line.back() == ';') {
                line.pop_back();
            }

            size_t param_open_pos = line.find('(');
            size_t first_space_pos = line.find(' ');

            if (param_open_pos != std::string::npos && (first_space_pos == std::string::npos || param_open_pos < first_space_pos)) {
                // Parameterized gate: e.g., rz(pi/4) q[0]
                command = line.substr(0, param_open_pos); // "rz"
                
                size_t param_close_pos = line.find(')');
                if (param_close_pos == std::string::npos) {
                    std::cerr << "Warning: Malformed QASM line (missing ')'): " << line << std::endl;
                    continue;
                }
                params = line.substr(param_open_pos + 1, param_close_pos - param_open_pos - 1); // "pi/4"
                
                size_t args_start_pos = line.find_first_not_of(' ', param_close_pos + 1);
                if (args_start_pos == std::string::npos) {
                    std::cerr << "Warning: Malformed QASM line (missing args): " << line << std::endl;
                    continue;
                }
                args = line.substr(args_start_pos); // "q[0]"

            } else if (first_space_pos != std::string::npos) {
                // Non-parameterized gate: e.g., x q[0]
                command = line.substr(0, first_space_pos); // "x"
                args = line.substr(first_space_pos + 1); // "q[0]"
            
            } else {
                // Command with no args: e.g., barrier
                command = line; // "barrier"
            }
            
            trim(command);
            trim(params);
            trim(args);
            
            // Apply the command (note the new signature)
            apply_qasm_command(command, params, args);
        }

        file.close();
    }
};


// ===================================================================
// == MAIN FUNCTION (Test)
// ===================================================================

int main_old(int argc, char* argv[]) {
    // Check for command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <qasm_filename.qasm>" << std::endl;
        return 1;
    }

    std::string qasm_filename = argv[1];
    const int NUM_SHOTS = 1 << 16; // 65,536 shots

    std::cout << "--- QASM File Test (CUDA C++) ---" << std::endl;
    std::cout << "Running on GPU..." << std::endl;

    try {
        // 1. & 2. Create Trajectories object
        // The new constructor handles scanning the file for qubit count
        // and register mapping.
        Trajectories circuit(qasm_filename, NUM_SHOTS);
        
        // 3. Run the circuit from the QASM file
        std::cout << "Running circuit from " << qasm_filename << "..." << std::endl;
        circuit.run_qasm_file(qasm_filename);
        
        // 4. Perform measurement
        std::cout << "\n--- Performing Measurement ---" << std::endl;
        std::map<std::string, int> measurement_results = circuit.measure_old();
        
        // 5. Print results
        double total_shots = static_cast<double>(NUM_SHOTS);

        // Get num_qubits from the object for formatting
        int num_qubits = circuit.get_num_qubits(); 

        for (const auto& pair : measurement_results) {
            double percentage = (pair.second / total_shots) * 100.0;
            std::cout << "  " << std::setw(num_qubits + 4) << std::left << pair.first
                      << ": " << std::setw(8) << pair.second << " occurrences"
                      << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    // When `circuit` goes out of scope, its destructor is called,
    // automatically freeing the GPU memory (d_data, d_rand_states).
    std::cout << "\n--- Test complete. Memory released via destructor. ---" << std::endl;
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Check for command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <qasm_filename.qasm>" << std::endl;
        return 1;
    }

    std::string qasm_filename = argv[1];
    const int NUM_SHOTS = 1 << 16; // 65,536 shots

    std::cout << "--- QASM File Test (CUDA C++) ---" << std::endl;
    std::cout << "Running on GPU..." << std::endl;

    try {
        // 1. & 2. Create Trajectories object
        Trajectories circuit(qasm_filename, NUM_SHOTS);
        
        // 3. Run the circuit from the QASM file
        std::cout << "Running circuit from " << qasm_filename << "..." << std::endl;
        circuit.run_qasm_file(qasm_filename);
        
        // 4. Perform measurement
        std::cout << "\n--- Performing Measurement ---" << std::endl;
        
        // MODIFIED: Return type is now <string, double>
        std::map<std::string, double> measurement_results = circuit.measure();
        
        // 5. Print results
        // MODIFIED: No longer need total_shots for calculation,
        // but it's good to know. We get num_qubits for formatting.
        int num_qubits = circuit.get_num_qubits(); 
        double prob_sum = 0.0; // To check normalization

        for (const auto& pair : measurement_results) {
            // MODIFIED: pair.second is now a probability
            double probability = pair.second;
            double percentage = probability * 100.0;
            prob_sum += probability;

            std::cout << "  " << std::setw(num_qubits + 4) << std::left << pair.first
                      << ": " << "Prob = " << std::setw(10) << std::fixed << std::setprecision(6) << probability
                      << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
        }

        std::cout << "---------------------------------" << std::endl;
        std::cout << "  Total Probability Sum: " << std::fixed << std::setprecision(6) << prob_sum << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- Test complete. Memory released via destructor. ---" << std::endl;
    
    return 0;
}