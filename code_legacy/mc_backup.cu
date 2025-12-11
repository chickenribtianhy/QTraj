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

// __global__ void sx_kernel(uint8_t* data, int num_shots, int num_bytes_per_shot, int qubit_index) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_shots) return;

//     uint8_t* my_shot_row = data + (size_t)tid * num_bytes_per_shot;
//     int current_state = get_qubit(my_shot_row, qubit_index);
//     // SX|0> = ((1+i)/2)|0> + ((1-i)/2)|1>
//     // SX|1> = ((1-i)/2)|0> + ((1+i)/2)|1>

// }

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
                          int qubit_index, double theta, double phi, double lambda, cuDoubleComplex global_phase) {
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
    weights[tid] = cuCmul(weights[tid], global_phase);
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

    // Use a stringstream to replace 'pi' with a parsable floating-point representation,
    // like M_PI, and then use the stream's built-in math capabilities.
    
    // 1. Normalize the string: Replace 'pi' with an identifier or a known number.
    // We'll replace 'pi' with a temporary constant and then use a mathematical expression evaluator.
    // For simplicity and dependency avoidance, let's stick to regex to parse the multiplier/divisor.

    // Regex to capture: [NUMERATOR][pi][/][DENOMINATOR]
    // Example: "3pi/4" -> mult=3/4
    // Example: "pi/4"  -> mult=1/4
    // Example: "2*pi"  -> mult=2
    std::regex re("([+-]?\\s*\\d*\\.?\\d*)\\s*\\*?\\s*pi\\s*\\/\\s*(\\d*\\.?\\d*)"); 
    std::smatch match;
    
    // --- Case 1: X * pi / Y (e.g., 3*pi/4, pi/4, 2pi/3) ---
    // The "pi/4" case from your old code is a simplified version of this.
    // Let's use string manipulation to handle "X/Y" * pi.
    size_t pi_pos = s.find("pi");
    if (pi_pos != std::string::npos) {
        std::string before_pi = s.substr(0, pi_pos);
        std::string after_pi = s.substr(pi_pos + 2); // 2 is length of "pi"
        
        trim(before_pi);
        trim(after_pi);
        
        double multiplier = 1.0;
        double divisor = 1.0;
        
        // Check for an explicit multiplier before 'pi' (e.g., "3*pi/4" or "3pi/4")
        if (!before_pi.empty()) {
            if (before_pi == "+") {
                multiplier = 1.0;
            } else if (before_pi == "-") {
                multiplier = -1.0;
            } else {
                if (before_pi.back() == '*') before_pi.pop_back();
                trim(before_pi);
                try { multiplier = std::stod(before_pi); }
                catch (...) { /* keep default */ }
            }
        }
        
        // Check for a divisor after 'pi' (e.g., "/4" in "3pi/4")
        if (after_pi.rfind('/', 0) == 0) {
            std::string denom_str = after_pi.substr(1);
            trim(denom_str);
            try { divisor = std::stod(denom_str); }
            catch (...) { 
                std::cerr << "Warning: Failed to parse pi denominator in '" << s << "'. Defaulting to 1." << std::endl;
                divisor = 1.0; 
            }
        }
        
        return multiplier * M_PI / divisor;
    }


    // --- Case 2: Simple numerical value (no 'pi') ---
    try { 
        return std::stod(s); 
    }
    catch (...) { 
        // This is only hit if it contains 'pi' but failed the above logic,
        // or it's a non-numeric string like 'a'.
        std::cerr << "Warning: Could not fully parse angle '" << s << "'. Defaulting to 0." << std::endl;
        return 0.0; 
    }
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

    void u3(int qubit_index, double theta, double phi, double lambda, cuDoubleComplex global_phase=make_cuDoubleComplex(1,0)) {
        u3_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, d_rand_states, d_weights, num_shots, num_bytes_per_shot, 
            qubit_index, theta, phi, lambda, global_phase
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
        // SX = U3(pi/2, -pi/2, pi/2) * e^(j*pi/4)
        // std::cout << "sxxxxx " << qubit_index << std::endl;
        this->u3(qubit_index, M_PI / 2.0, -M_PI / 2.0, M_PI / 2.0, make_cuDoubleComplex(cos(M_PI/4.0), sin(M_PI/4.0)));
    }

    /**
     * @brief Rz-gate (Z-axis rotation)
     * QASM: rz(lambda) q[i]
     */
    void rz(int qubit_index, double lambda) {
        // Rz = U3(0, 0, lambda) (up to global phase)
        // In this probabilistic simulator, theta=0 means no change
        // in the |0>/|1> probabilities, which is correct.
        // std::cout<<"rzzzzzzzz" << std::endl;
        // std::cout<<"lambda "<< lambda << std::endl;

        // double cos_theta_half = cos(0 / 2.0);
        // double sin_theta_half = sin(0 / 2.0);

        // double prob0_to_1 = sin_theta_half * sin_theta_half;
        // double prob1_to_1 = cos_theta_half * cos_theta_half;

        // cuDoubleComplex exp_lambda = make_cuDoubleComplex(cos(lambda), sin(lambda));
        // cuDoubleComplex exp_phi = make_cuDoubleComplex(cos(0), sin(0));
        // cuDoubleComplex exp_phi_lambda = make_cuDoubleComplex(cos(lambda+0), sin(lambda+0));

        // // m00 = cos(t/2)
        // cuDoubleComplex m00 = make_cuDoubleComplex(cos_theta_half, 0.0);
        // // m01 = -e^(i*l) * sin(t/2)
        // cuDoubleComplex m01 = cuCmul(make_cuDoubleComplex(-sin_theta_half, 0.0), exp_lambda);
        // // m10 = e^(i*p) * sin(t/2)
        // cuDoubleComplex m10 = cuCmul(make_cuDoubleComplex(sin_theta_half, 0.0), exp_phi);
        // // m11 = e^(i*(p+l)) * cos(t/2)
        // cuDoubleComplex m11 = cuCmul(make_cuDoubleComplex(cos_theta_half, 0.0), exp_phi_lambda);
        
        // auto printc = [](const char* name, cuDoubleComplex z){
        //     std::cout << name << " = "
        //             << std::setprecision(17)
        //             << cuCreal(z) << (cuCimag(z) >= 0 ? " + " : " - ")
        //             << std::abs(cuCimag(z)) << "i\n";
        // };

        // printc("exp_lambda", exp_lambda);
        // printc("exp_phi", exp_phi);
        // printc("exp_phi_lambda", exp_phi_lambda);
        // printc("m00", m00);
        // printc("m01", m01);
        // printc("m10", m10);
        // printc("m11", m11);
        
        this->u3(qubit_index, 0.0, 0.0, lambda, make_cuDoubleComplex(cos(-lambda/2), sin(-lambda/2)));
    }


    // --- Measurement ---

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

    // Returns a map of <state, probability>
    std::map<std::string, double> measure() {
        // Synchronize device to make sure all kernels are finished before measuring
        checkCuda( cudaDeviceSynchronize() );

        if (num_bytes_per_shot > 8) {
            throw std::runtime_error("Measure() for > 64 qubits is not implemented.");
        }

        // 1. Create Thrust device vectors for intermediate/output storage
        thrust::device_vector<uint64_t> d_hashes(num_shots);
        thrust::device_vector<uint64_t> d_unique_hashes(num_shots);
        thrust::device_vector<cuDoubleComplex> d_summed_weights(num_shots);

        // 2. Launch kernel to pack all byte rows into uint64_t hashes
        pack_to_uint64_kernel<<<getGridDim(), getBlockDim()>>>(
            d_data, thrust::raw_pointer_cast(d_hashes.data()), 
            num_shots, num_bytes_per_shot
        );
        checkCuda( cudaPeekAtLastError() );

        // 3. Wrap the existing raw device pointers for Thrust
        thrust::device_ptr<cuDoubleComplex> d_weights_ptr(d_weights);

        // 4. Sort by key (the hashes)
        thrust::sort_by_key(
            d_hashes.begin(), d_hashes.end(),   // Keys to sort by
            d_weights_ptr                       // Values to sort in parallel
        );

        
/*          // ==========================================================
        // == START: DEBUG PRINT 1 (BEFORE REDUCTION)              ==
        // ==========================================================
        // Copy the FULL sorted arrays from GPU (device) to CPU (host)
        std::vector<uint64_t> h_sorted_hashes(num_shots);
        std::vector<cuDoubleComplex> h_sorted_weights(num_shots);
        
        // Copy hashes (from device_vector)
        thrust::copy(d_hashes.begin(), d_hashes.end(), h_sorted_hashes.begin());
        
        // --- THIS IS THE FIX ---
        // d_weights_ptr is the "begin" iterator
        // d_weights_ptr + num_shots is the "end" iterator
        thrust::copy(d_weights_ptr, d_weights_ptr + num_shots, h_sorted_weights.begin());
        // --- END FIX ---
        
        std::cout << "\n--- Debug: Raw Sorted Hashes & Weights (Pre-Reduction) ---" << std::endl;
        std::cout << std::fixed << std::setprecision(6); // Set precision for complex numbers

        uint64_t last_hash = (num_shots > 0) ? h_sorted_hashes[0] : 0;
        int print_limit = 100; // Only print the first 100 entries

        for (int i = 0; i < num_shots; ++i) {
            // Check if this is a new hash group
            if (h_sorted_hashes[i] != last_hash) {
                if (i < print_limit) { // Only print group breaks if we are still printing
                    std::cout << "  --- Group break ---" << std::endl;
                }
                last_hash = h_sorted_hashes[i];
            }

            if (i < print_limit) { 
                std::cout << "  i=" << std::setw(5) << i
                          << "  Hash: " << std::setw(10) << format_ket(h_sorted_hashes[i])
                          << "  Raw Weight: (" << std::setw(10) << h_sorted_weights[i].x
                          << ", " << std::setw(10) << h_sorted_weights[i].y << ")" << std::endl;
            } else if (i == print_limit) {
                std::cout << "  ... (skipping " << (num_shots - print_limit) << " more raw entries) ..." << std::endl;
            }
        }
        std::cout << "--- End Debug Print 1 ---\n" << std::endl;
        // ========================================================
        // == END: DEBUG PRINT 1                                 ==
        // ======================================================== 
 */
        // 5. Reduce by key (the "unique" and "sum" operation)
        auto end = thrust::reduce_by_key(
            d_hashes.begin(), d_hashes.end(),           // Input keys (sorted hashes)
            d_weights_ptr,                             // Input values (sorted weights)
            d_unique_hashes.begin(),                   // Output keys (unique hashes)
            d_summed_weights.begin(),                  // Output values (SUMMED weights)
            thrust::equal_to<uint64_t>(),              // How to compare keys
            cuDoubleComplex_add()                      // How to sum values
        );

        // `end.first` points to the end of the unique hashes
        int num_unique_states = end.first - d_unique_hashes.begin();
        
        // 6. Copy reduction results from GPU to CPU
        std::vector<uint64_t> h_unique_hashes(num_unique_states);
        std::vector<cuDoubleComplex> h_summed_weights(num_unique_states);

        thrust::copy(d_unique_hashes.begin(), d_unique_hashes.begin() + num_unique_states, h_unique_hashes.begin());
        thrust::copy(d_summed_weights.begin(), d_summed_weights.begin() + num_unique_states, h_summed_weights.begin());
        

/*         // ==========================================================
        // == START: DEBUG PRINT 2 (AFTER REDUCTION)               ==
        // ==========================================================
        std::cout << "\n--- Debug: Unique Hashes and Summed Weights (Post-Reduction) ---" << std::endl;
        std::cout << "Found " << num_unique_states << " unique states." << std::endl;
        std::cout << std::fixed << std::setprecision(6); // Set precision

        for (int i = 0; i < num_unique_states; ++i) {
            std::cout << "  i=" << std::setw(4) << i
                      << "  Hash: " << std::setw(10) << format_ket(h_unique_hashes[i])
                      << "  Summed Weight: (" << std::setw(10) << h_summed_weights[i].x
                      << ", " << std::setw(10) << h_summed_weights[i].y << ")" << std::endl;
        }
        std::cout << "--- End Debug Print 2 ---\n" << std::endl;
        // ========================================================
        // == END: DEBUG PRINT 2                                 ==
        // ========================================================
 */

        // 7. Format results into the final map on the CPU
        // WARNING: double can only hold 1<<53, so for qubit number >= 54, implementation needed
        std::map<std::string, double> results;
        std::map<std::string, double> counts;
        double total_effective_shots = 0;
        for (int i = 0; i < num_unique_states; ++i) {
            std::string ket = format_ket(h_unique_hashes[i]);
            cuDoubleComplex W_k_total = h_summed_weights[i];
            
            // This is the calculation you had in your last snippet:
            // |Sum(Weights)|^2
            double effective_shots = cuCabs(W_k_total) * cuCabs(W_k_total);
            counts[ket] = effective_shots;
            total_effective_shots += effective_shots;
        }
        for (int i = 0; i < num_unique_states; ++i) {
            std::string ket = format_ket(h_unique_hashes[i]);
            results[ket] = counts[ket] / total_effective_shots;
            // std::cout << counts[ket] << " / " << total_effective_shots << " = " << results[ket] << std::endl;
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
int main(int argc, char* argv[]) {
    // Check for command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <qasm_filename.qasm>" << std::endl;
        return 1;
    }

    std::string qasm_filename = argv[1];
    const int NUM_SHOTS = 1 << 28; // 65,536 shots
    // const int NUM_SHOTS = 10; 

    std::cout << "--- QASM File Test (CUDA C++) ---" << std::endl;
    std::cout << "Running on GPU..." << std::endl;

    try {
        // 1. & 2. Create Trajectories object
        Trajectories circuit(qasm_filename, NUM_SHOTS);
        
        // 3. Run the circuit from the QASM file
        std::cout << "Running circuit from " << qasm_filename << "..." << std::endl;
        circuit.run_qasm_file(qasm_filename);
        
        // 4. Perform measurement
        std::cout << "\n--- Performing Measurement (Top 10 Results) ---" << std::endl;
        
        // Get the map (which is sorted by key: "|00>", "|01>", etc.)
        std::map<std::string, double> measurement_results = circuit.measure();
        
        // 5. --- SORTING LOGIC ---
        
        // 5a. Copy the map into a vector of pairs
        std::vector<std::pair<std::string, double>> sorted_results(
            measurement_results.begin(),
            measurement_results.end()
        );

        // 5b. Define a comparator to sort by value (the double) in descending order
        auto comparator = [](const std::pair<std::string, double>& a, 
                             const std::pair<std::string, double>& b) {
            return a.second > b.second; // '>' means highest probability first
        };

        // 5c. Sort the vector using the comparator
        std::sort(sorted_results.begin(), sorted_results.end(), comparator);

        // 6. --- PRINT FROM SORTED VECTOR (TOP 10 ONLY) ---
        
        int num_qubits = circuit.get_num_qubits(); 
        double prob_sum = 0.0; // We will sum all probabilities, even if not printed

        // Loop over the *sorted_results vector*
        for (size_t i = 0; i < sorted_results.size(); ++i) {
            const auto& pair = sorted_results[i];
            
            double probability = pair.second;
            double percentage = probability * 100.0;
            
            // Add to total sum regardless of printing
            prob_sum += probability;

            // --- THIS IS THE CHANGE ---
            // Only print if i is less than 10
            if (i < 10) {
                std::cout << "  " << std::setw(num_qubits + 4) << std::left << pair.first
                          << ": " << "Prob = " << std::setw(10) << std::fixed << std::setprecision(6) << probability
                          << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
            }
            // --- END CHANGE ---
        }

        if (sorted_results.size() > 10) {
            std::cout << "  ... (and " << (sorted_results.size() - 10) << " more states)" << std::endl;
        }

        std::cout << "---------------------------------" << std::endl;
        std::cout << "  Total Probability Sum (all states): " << std::fixed << std::setprecision(6) << prob_sum << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- Test complete. Memory released via destructor. ---" << std::endl;
    
    return 0;
}