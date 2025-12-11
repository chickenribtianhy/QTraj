import time
import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


DEVICE = "GPU"
METHOD = "statevector"
SHOTS = 1<<15

def load_circuit_from_qasm(qasm_file_path):
    """
    Loads a QuantumCircuit object from a QASM file.

    Args:
        qasm_file_path (str): The path to the .qasm file.

    Returns:
        QuantumCircuit: The loaded quantum circuit.
    """
    try:
        circuit = QuantumCircuit.from_qasm_file(qasm_file_path)
        print(f"Successfully loaded circuit from '{qasm_file_path}'")
        print(f"Circuit details: {circuit.num_qubits} qubits, {circuit.depth()} depth")
        return circuit
    except Exception as e:
        print(f"Error loading QASM file '{qasm_file_path}': {e}")
        sys.exit(1)

def benchmark_circuit(circuit, shots=1024):
    """
    Benchmarks the simulation of a given quantum circuit using qiskit_aer.

    Args:
        circuit (QuantumCircuit): The circuit to simulate.
        shots (int): The number of times to run the simulation.

    Returns:
        float: The execution time in seconds.
    """
    print(f"\nStarting benchmark with {shots} shots...")
    
    # Initialize the Aer simulator
    # 'automatic' will choose the best method.
    simulator = AerSimulator(method=METHOD, device=DEVICE)

    # --- THIS IS THE CORRECTED PART ---
    # In Qiskit 1.0+, transpilation must be done explicitly
    # using the transpile() function.
    print("Transpiling circuit for the simulator...")
    transpiled_circuit = transpile(circuit, simulator)
    # --- END OF CORRECTION ---

    # Start the timer
    start_time = time.time()

    # Run the simulation
    # The run() method now expects a transpiled circuit.
    job = simulator.run(transpiled_circuit, shots=shots)
    end_time = time.time()
    comp_time = end_time - start_time

    result = job.result() # Wait for the job to finish

    # Stop the timer
    end_time2 = time.time()

    meas_time = end_time2 - end_time

    print("Benchmark finished.")
    print(f"Result status: {result.status}")
    print()
    print(result.get_counts())
    
    return comp_time, meas_time

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) > 1:
        qasm_file = sys.argv[1]
    else:
        # Use the default example file if no argument is given
        qasm_file = "example.qasm"
        print(f"No QASM file provided. Using default '{qasm_file}'")

    # Load the circuit
    quantum_circuit = load_circuit_from_qasm(qasm_file)

    # Run the benchmark
    comp_time, meas_time = benchmark_circuit(quantum_circuit, shots=SHOTS)
    # for test
    statevector = Statevector(quantum_circuit)
    print(statevector)
    for i, _ in enumerate(statevector):
        if i >= 20:
            break
        print(_)

    # print(f"\n--- Benchmark Summary ---")
    # print(f"File:                   {qasm_file}")
    # print(f"Device Used:            {DEVICE}")
    # print(f"Method Used:            {METHOD}")
    # print(f"Shots:                  {SHOTS}")
    
    # if comp_time is not None and meas_time is not None:
    #     print(f"Comp. Time:             {comp_time:.6f} seconds")
    #     print(f"Meas. Time:             {meas_time:.6f} seconds")
    #     print(f"Total (Comp + Meas):    {(comp_time + meas_time):.6f} seconds")
    # else:
    #     print("\nBenchmark failed.")