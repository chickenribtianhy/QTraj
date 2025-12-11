import cmath
import math
import random
import numpy as np
from collections import OrderedDict

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy found! GPU acceleration is enabled.")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found. Running on CPU with NumPy.")

# For demonstration, you can force CPU or GPU
# GPU_AVAILABLE = False
if GPU_AVAILABLE:
    xp = cp
else:
    xp = np

NUM_TRAJECTORIES = int(1<<10)

class Trajectories:

    def __init__(self, num_qubits, shots=NUM_TRAJECTORIES):
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        
        self._num_qubits = num_qubits
        self.data = xp.zeros((shots, num_qubits), dtype=bool)

    def measure(self):
        """
        Counts the number of occurrences of each quantum basis state.

        Returns:
            OrderedDict: A dictionary mapping the ket string (e.g., "|011>") 
                         to its integer count, sorted by most common state.
        """
        # 1. Ensure data is on the CPU for processing
        if GPU_AVAILABLE:
            cpu_data = self.data.get()
        else:
            cpu_data = self.data

        # 2. Convert boolean rows to unique integers for efficient counting
        powers_of_2 = 2 ** np.arange(self._num_qubits - 1, -1, -1, dtype=np.int64)
        kets_as_integers = cpu_data.astype(np.int64) @ powers_of_2
        
        # 3. Find unique states and count their occurrences
        unique_kets, counts = np.unique(kets_as_integers, return_counts=True)
        
        # 4. Sort by count in descending order
        sort_indices = np.argsort(counts)[::-1]
        
        # 5. Build the result dictionary
        results = OrderedDict()
        for i in sort_indices:
            ket_int = unique_kets[i]
            count = counts[i]
            ket_label = format(ket_int, f'0{self._num_qubits}b')
            results[f"|{ket_label}>"] = count
            
        return results

    def __str__(self, max_states_to_show=10):
        """
        Provides a statistical ket representation of the quantum state by calling measure().
        """
        counts_dict = self.measure()
        return str(counts_dict)
    

    def __repr__(self):
        return self.__str__()

    def branch(self, probs):
        rand = xp.random.random(probs.shape)
        return rand < probs

    def x(self, qubit_index):
        self.data[:, qubit_index] = ~self.data[:, qubit_index]

    def hadamard(self, qubit_index):
        self.u3(qubit_index, theta=math.pi / 2, phi=0, lambda_=math.pi)

    def u3(self, qubit_index, theta=0, phi=0, lambda_=0):
        cos_theta_half = xp.cos(theta / 2)
        sin_theta_half = xp.sin(theta / 2)
        prob0_to_1 = sin_theta_half**2
        prob1_to_1 = cos_theta_half**2
        current_states = self.data[:, qubit_index]
        probs = xp.where(current_states, prob1_to_1, prob0_to_1)
        self.data[:, qubit_index] = self.branch(probs)

    def cx(self, control_index, target_index):
        self.data[:, target_index] ^= self.data[:, control_index]
    
    def cu3(self, control_index, target_index, theta=0, phi=0, lambda_=0):
        control_states = self.data[:, control_index]
        target_states = self.data[:, target_index]
        cos_theta_half = xp.cos(theta / 2)
        sin_theta_half = xp.sin(theta / 2)
        prob0_to_1 = sin_theta_half**2
        prob1_to_1 = cos_theta_half**2
        u3_probs = xp.where(target_states, prob1_to_1, prob0_to_1)
        identity_probs = target_states.astype(xp.float32)
        final_probs = xp.where(control_states, u3_probs, identity_probs)
        self.data[:, target_index] = self.branch(final_probs)
    
    def cp(self, control_index, target_index, lambda_):
        self.cu3(control_index, target_index, theta=0, phi=0, lambda_=lambda_)

if __name__ == "__main__":
    # --- Example: Create a 3-qubit GHZ state and measure it ---
    
    print(f"--- GHZ State Test ---")
    print(f"Running on {'GPU' if GPU_AVAILABLE else 'CPU'}")
    
    ghz = Trajectories(num_qubits=3)
    
    # 1. Start in state |000>
    print("\nInitial state:")
    print(ghz)

    # 2. Build the entangled state
    ghz.hadamard(0)
    ghz.cx(0, 1)
    ghz.cx(0, 2)
    
    # 3. Show the final state representation
    print("\nFinal GHZ State:")
    print(ghz)
    
    # 4. Perform the measurement
    print("\n--- Performing Measurement ---")
    measurement_results = ghz.measure()
    
    print("Measurement Counts:")
    # Print the results in a nice format
    for ket, count in measurement_results.items():
        print(f"  {ket}: {count} occurrences")