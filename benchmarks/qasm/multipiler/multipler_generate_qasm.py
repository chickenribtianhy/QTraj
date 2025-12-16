import math
import random
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import qasm2  # Required for Qiskit 1.0+

def generate_circuit_for_n(target_n):
    """
    Generates a valid OpenQASM 2.0 circuit using exactly target_n qubits.
    Fits a multiplier logic if n >= 4, otherwise generates dummy gates.
    """
    
    # 1. Setup Registers
    # Using a single register 'q' to match your requested format
    q = QuantumRegister(target_n, 'q')
    c = ClassicalRegister(target_n, 'c')
    qc = QuantumCircuit(q, c)
    
    # 2. Determine Logic based on size
    if target_n < 4:
        # Too small for multiplier logic
        qc.x(q[0])
        if target_n > 1:
            qc.h(q[1])
        if target_n > 2:
            qc.cx(q[0], q[2])
            
    else:
        # Calculate max possible input size
        # Formula: n_total = n_a + n_b + n_out
        # We try to use half the qubits for input, half for output
        total_input_bits = target_n // 2
        
        # Split inputs between A and B
        n_a = total_input_bits // 2
        n_b = total_input_bits - n_a
        
        # Ensure at least 1 bit each
        if n_a < 1: n_a = 1
        if n_b < 1: n_b = 1
        
        # Output starts after inputs
        idx_a_start = 0
        idx_b_start = n_a
        idx_out_start = n_a + n_b
        
        # Generate Random Inputs
        max_a = (2 ** n_a) - 1
        max_b = (2 ** n_b) - 1
        
        val_a = random.randint(0, max_a)
        val_b = random.randint(0, max_b)
        
        # -- ENCODE INPUTS (X Gates) --
        # Convert to binary and reverse for Little Endian
        bin_a = format(val_a, f'0{n_a}b')[::-1]
        bin_b = format(val_b, f'0{n_b}b')[::-1]

        for i, char in enumerate(bin_a):
            if char == '1': qc.x(q[idx_a_start + i])
        for i, char in enumerate(bin_b):
            if char == '1': qc.x(q[idx_b_start + i])

        qc.barrier()
        
        # -- MULTIPLIER LOGIC (Shift and Add) --
        # We use CCX (Toffoli) to add partial products
        for i in range(n_a):
            for j in range(n_b):
                target_idx = idx_out_start + i + j
                # Ensure we don't write past the end of the register
                if target_idx < target_n:
                    qc.ccx(q[idx_a_start + i], q[idx_b_start + j], q[target_idx])

        qc.barrier()

    # 3. Measure All
    for i in range(target_n):
        qc.measure(q[i], c[i])

    # 4. Define Basis Gates
    # Note: 'barrier', 'measure', 'id' are excluded from basis_gates in Qiskit 1.0+
    custom_basis = [
        'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx',
        'rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3',
        'cx', 'cy', 'cz', 'ch', 'swap',
        'cu3', 'cp', 'cu1', 'crz',
        'ccx'
    ]

    # 5. Transpile & Export
    try:
        transpiled_qc = transpile(qc, basis_gates=custom_basis, optimization_level=1)
        return qasm2.dumps(transpiled_qc)
    except Exception as e:
        print(f"Error transpiling for n={target_n}: {e}")
        return qasm2.dumps(qc)

# --- Main Execution Loop ---
print("Generating circuits in current directory...")

for n in range(1, 101):
    qasm_content = generate_circuit_for_n(n)
    filename = f"multiplier_n{n}.qasm"
    
    with open(filename, "w") as f:
        f.write(qasm_content)
        
    print(f"Generated: {filename}")

print("\nDone.")