import time, sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

DEVICE = "GPU"
METHOD = "statevector"
SHOTS = 1 << 15

def bitstring_to_index(bitstr: str) -> int:
    # bitstr is written as |q_{n-1} ... q_0> (q0 rightmost)
    return int(bitstr, 2)

def load_circuit_from_qasm(qasm_file_path):
    try:
        circuit = QuantumCircuit.from_qasm_file(qasm_file_path)
        print(f"Successfully loaded circuit from '{qasm_file_path}'")
        print(f"Circuit details: {circuit.num_qubits} qubits, {circuit.depth()} depth")
        return circuit
    except Exception as e:
        print(f"Error loading QASM file '{qasm_file_path}': {e}")
        sys.exit(1)

def benchmark_circuit_only_selected_bases(circuit, want_bitstrings):
    simulator = AerSimulator(method=METHOD, device=DEVICE)
    transpiled = transpile(circuit, simulator)

    # Convert requested basis bitstrings -> indices
    want_indices = [bitstring_to_index(b) for b in want_bitstrings]
    print(want_indices)

    # Ask Aer to return ONLY these amplitudes (+ probs if you want)
    # These are Aer "save" instructions; no full statevector is copied back.
    transpiled.save_amplitudes(want_indices, label="amps")
    transpiled.save_amplitudes_squared(want_indices, label="probs")

    t0 = time.time()
    # Shots are irrelevant here (no measurement sampling needed), so keep it 1.
    job = simulator.run(transpiled, shots=1<<10)
    result = job.result()
    t1 = time.time()

    data = result.data(0)
    amps = data["amps"]
    probs = data["probs"]
    print(amps)
    print(probs)

    print("\nSelected basis outputs (only):")
    for b, a, p in zip(want_bitstrings, amps, probs):
        # 'a' is complex, 'p' is float
        print(f"|{b}>: amp={a}  prob={p}")

    return (t1 - t0)

if __name__ == "__main__":
    qasm_file = sys.argv[1] if len(sys.argv) > 1 else "example.qasm"
    qc = load_circuit_from_qasm(qasm_file)

    # Example: only these basis states
    want = [
        "0011111100000000100011111110",
        "0011111100000000000011111110",
        # # "1111000000000000111111111110",
        # "1111000000000000111111111110"
    ]

    sim_time = benchmark_circuit_only_selected_bases(qc, want)
    print(f"\nSim time (save subset): {sim_time:.6f} seconds")