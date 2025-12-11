import time
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Config
DEVICE = "GPU"  # Change to "CPU" if you don't have a CUDA GPU
METHOD = "statevector" 
SHOTS = 1 

def load_circuit_from_qasm(qasm_file_path):
    try:
        circuit = QuantumCircuit.from_qasm_file(qasm_file_path)
        print(f"Successfully loaded circuit from '{qasm_file_path}'")
        print(f"Circuit details: {circuit.num_qubits} qubits, {circuit.depth()} depth")
        return circuit
    except Exception as e:
        print(f"Error loading QASM file '{qasm_file_path}': {e}")
        sys.exit(1)

def benchmark_circuit(circuit):
    print(f"\nStarting benchmark (Method: {METHOD}, Device: {DEVICE})...")
    
    # 1. Remove measurements
    print("Removing final measurements for Statevector analysis...")
    circuit.remove_final_measurements()

    # 2. Setup Simulator
    simulator = AerSimulator(method=METHOD, device=DEVICE)

    # 3. Add 'save_statevector'
    # Since we are NOT transpiling, we append this instruction directly
    # to the original circuit object.
    circuit.save_statevector()

    # --- SKIPPING TRANSPILATION ---
    # AerSimulator supports almost all QASM gates natively.
    # We pass the raw circuit directly to run().
    print("Skipping transpilation (running raw circuit)...")

    # 4. Run Simulation
    start_time = time.time()
    
    # Pass 'circuit' directly instead of 'transpiled_circuit'
    job = simulator.run(circuit, shots=SHOTS)
    
    result = job.result()
    end_time = time.time()
    comp_time = end_time - start_time

    # 5. Extract the Statevector
    if result.success:
        # You must pass the SAME circuit object used in run() to retrieve the data
        sv = result.get_statevector(circuit)
        
        coeffs = sv.to_dict()

        print("Benchmark finished.")
        print(f"Result status: {result.status}")
        print("-" * 50)
        print("COMPLEX COEFFICIENTS (Showing non-zero states):")
        
        count = 0
        for state_key, amplitude in coeffs.items():
            if abs(amplitude) > 1e-10:
                prob = abs(amplitude)**2
                print(f"State: {state_key}")
                print(f"  Amplitude:   {amplitude}")
                print(f"  Probability: {prob:.6f}")
                print("-" * 30)
                count += 1
                if count >= 20: 
                    print("... (Truncating output after 20 states) ...")
                    break
    else:
        print(f"Simulation failed: {result.status}")
        return None

    return comp_time

if __name__ == "__main__":
    if len(sys.argv) > 1:
        qasm_file = sys.argv[1]
    else:
        qasm_file = "example.qasm"
        print(f"No QASM file provided. Using default '{qasm_file}'")

    quantum_circuit = load_circuit_from_qasm(qasm_file)
    comp_time = benchmark_circuit(quantum_circuit)

    print(f"\n--- Benchmark Summary ---")
    print(f"File:                   {qasm_file}")
    print(f"Device Used:            {DEVICE}")
    print(f"Method Used:            {METHOD}")
    
    if comp_time is not None:
        print(f"Computation Time:       {comp_time:.6f} seconds")
    else:
        print("\nBenchmark failed.")