import cmath
import math
import random
import numpy as np
from collections import OrderedDict
import gc 

try:
    import cupy as cp
    GPU_AVAILABLE = True
    # print("CuPy found! GPU acceleration is enabled.")
except ImportError:
    GPU_AVAILABLE = False
    # print("CuPy not found. Running on CPU with NumPy.")

if GPU_AVAILABLE:
    xp = cp
else:
    xp = np

NUM_TRAJECTORIES = 1<<8

class Trajectories:

    def __init__(self, num_qubits, shots=NUM_TRAJECTORIES):
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        
        self._num_qubits = num_qubits
        self.data = xp.zeros((shots, (num_qubits + 7) // 8), dtype=xp.uint8)
        # print("Trajectories object created.")

    # --- NEW: Methods for releasing memory ---
    def release(self):
        """
        Explicitly releases the GPU memory held by this object by clearing
        the CuPy memory pool. This is the recommended way to free memory.
        """
        if GPU_AVAILABLE and hasattr(self, 'data') and self.data is not None:
            self.data = None
            mempool = xp.get_default_memory_pool()
            mempool.free_all_blocks()
            # print("GPU memory has been explicitly released.")

    def __del__(self):
        """
        Destructor that attempts to free GPU memory upon object garbage collection.
        It's safer to call .release() manually.
        """
        # This check is important because at shutdown, `xp` or other modules
        # might already be gone, causing errors.
        try:
            if GPU_AVAILABLE and hasattr(self, 'data') and self.data is not None:
                # print("Trajectories object deconstructed. Attempting to release memory...")
                self.release()
        except (NameError, AttributeError):
            # Fail silently if modules are already garbage-collected
            pass

    def _get_qubit(self, qubit_index):
        byte_index = qubit_index // 8
        bit_index = qubit_index % 8
        mask = 1 << bit_index
        # byte_column = self.data[:, byte_index]
        return (self.data[:, byte_index] & mask) >> bit_index

    def _set_qubit(self, qubit_index, new_states):
        byte_index = qubit_index // 8
        bit_index = qubit_index % 8
        # print(f"bit index is {bit_index}, byte index is {byte_index}")
        mask = 1 << bit_index
        # print(f"mask is {mask}")
        shifted_new_states = (new_states.astype(xp.uint8) << bit_index)
        # print(f"shifted_new_states is {shifted_new_states}")
        self.data[:, byte_index] = (self.data[:, byte_index] & ~mask) | shifted_new_states
        # print(f"new state is {self.data}")

    def measure(self):
        if self.data is None:
            print("Memory has been released. Cannot measure.")
            return None

        # Calculate the number of bytes needed for the given number of qubits
        num_bytes = (self._num_qubits + 7) // 8

        # --- HASHING LOGIC to find unique_rows and counts ---
        if num_bytes <= 8:
            # FAST PATH: Hashing for <= 64 Qubits

            # 1. Pad data to 8 bytes so it can be viewed as uint64
            padded_data = xp.zeros((self.data.shape[0], 8), dtype=xp.uint8)
            padded_data[:, -num_bytes:] = self.data[:, ::-1]

            # 2. View the 8-byte rows as single uint64 integers (the "hashes")
            hashes = padded_data.view(xp.uint64).flatten()

            # 3. Run unique on the 1D integer array. This is much faster.
            unique_hashes, counts = xp.unique(hashes, return_counts=True)

            # 4. Convert the unique hashes back into byte rows for the unpacking step
            unique_rows_padded = unique_hashes.reshape(-1, 1).view(xp.uint8)
            unique_rows = unique_rows_padded[:, -num_bytes:]
        else:
            # FALLBACK PATH: Sorting for > 64 Qubits
            print("Warning: Using slower sort-based measurement for > 64 qubits.")
            unique_rows, counts = xp.unique(self.data, axis=0, return_counts=True)
        
        # --- GPU UNPACKING LOGIC (Unchanged as requested) ---

        # 2. Unpack all unique rows at once on the GPU using the flatten-reshape workaround
        if unique_rows.shape[0] > 0: # Ensure not trying to reshape an empty array
            unpacked_bits_gpu = xp.unpackbits(unique_rows.flatten()).reshape(unique_rows.shape[0], -1)
        else:
            unpacked_bits_gpu = xp.array([], dtype=xp.uint8).reshape(0, self._num_qubits)

        # 3. Slice to the actual number of qubits on the GPU
        actual_bits_gpu = unpacked_bits_gpu[:, -self._num_qubits:]
        
        # --- Data transfer from GPU to CPU ---
        if GPU_AVAILABLE:
            actual_bits = actual_bits_gpu.get()
            counts = counts.get()
        else:
            actual_bits = actual_bits_gpu

        # 4. Format results into the final dictionary on the CPU
        result = {}
        for bit_row, count in zip(actual_bits, counts):
            ket_label = "".join(map(str, bit_row))
            result[f"|{ket_label}>"] = count
            
        return result
    
    def __str__(self, max_states_to_show=10):
        counts_dict = self.measure()
        return str(counts_dict)

    def __repr__(self):
        return self.__str__()

    def branch(self, probs):
        rand = xp.random.random(probs.shape, dtype=xp.float32)
        return rand < probs

    def x(self, qubit_index):
        self._set_qubit(qubit_index, 1 - self._get_qubit(qubit_index))

    def h(self, qubit_index):
        self.u3(qubit_index, theta=math.pi / 2, phi=0, lambda_=math.pi)

    def u3(self, qubit_index, theta=0, phi=0, lambda_=0):
        # print(f"qubit index is {qubit_index}")
        cos_theta_half = xp.cos(theta / 2)
        sin_theta_half = xp.sin(theta / 2)
        prob0_to_1 = sin_theta_half**2
        prob1_to_1 = cos_theta_half**2
        current_states = self._get_qubit(qubit_index)
        probs = xp.where(current_states, prob1_to_1, prob0_to_1)
        self._set_qubit(qubit_index, self.branch(probs))

    def cx(self, control_index, target_index):
        self._set_qubit(target_index, self._get_qubit(control_index) ^ self._get_qubit(target_index))

    def cu3(self, control_index, target_index, theta=0, phi=0, lambda_=0):
        control_states = self._get_qubit(control_index)
        target_states = self._get_qubit(target_index)
        prob0_to_1 = xp.sin(theta / 2)**2
        prob1_to_1 = xp.cos(theta / 2)**2
        u3_probs = xp.where(target_states, prob1_to_1, prob0_to_1)
        identity_probs = target_states.astype(xp.float16)
        final_probs = xp.where(control_states, u3_probs, identity_probs)
        self._set_qubit(target_index, self.branch(final_probs))
    
    def cp(self, control_index, target_index, lambda_):
        self.cu3(control_index, target_index, theta=0, phi=0, lambda_=lambda_)


if __name__ == "__main__":
    print(f"--- GHZ State Test ---")
    print(f"Running on {'GPU' if GPU_AVAILABLE else 'CPU'}")
    
    ghz = Trajectories(num_qubits=3)
    
    ghz.hadamard(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    
    print("\n--- Performing Measurement ---")
    measurement_results = ghz.measure()
    
    total_shots = sum(measurement_results.values())
    for ket, count in measurement_results.items():
        percentage = (count / total_shots) * 100
        print(f"  {ket}: {count} occurrences ({percentage:.2f}%)")

    print("\n--- Releasing Memory ---")
    # Recommended way: Call release() explicitly
    # ghz.release()
    
    # Alternatively, to see the destructor work, we delete the reference
    # and call the garbage collector to force cleanup for this demo.
    print("Deleting ghz object reference...")
    del ghz
    print("Forcing garbage collection...")
    gc.collect()