import math
import random
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import qasm2

def generate_adder_for_n(target_n):
    """
    Generates a Quantum Ripple Carry Adder circuit using exactly target_n qubits.
    """
    
    # 1. Setup Registers
    q = QuantumRegister(target_n, 'q')
    c = ClassicalRegister(target_n, 'c')
    qc = QuantumCircuit(q, c)
    
    # 2. Logic Determination
    # A Full Adder requires: Input A + Input B + Carry Chain.
    # We roughly need 3 qubits per bit of precision (A_i, B_i, Carry_i).
    # Formula: 3 * bits <= target_n
    
    if target_n < 3:
        # Too small for a real full adder
        # Just generate dummy gates to satisfy file requirement
        qc.x(q[0])
        if target_n > 1:
            qc.cx(q[0], q[1])
            
    else:
        # Calculate bit width (w)
        # We need A(w), B(w), and Carry(w+1) ideally, or just Carry(w).
        # We allocate roughly target_n // 3 for the width.
        w = target_n // 3
        if w < 1: w = 1
        
        # Define indices
        # A occupies [0 ... w-1]
        # B occupies [w ... 2w-1]
        # Carries occupy [2w ... 2w + w] (approx)
        idx_a = 0
        idx_b = w
        idx_c = 2 * w 
        
        # Generate Random Inputs
        max_val = (2 ** w) - 1
        val_a = random.randint(0, max_val)
        val_b = random.randint(0, max_val)
        
        # -- ENCODE INPUTS --
        bin_a = format(val_a, f'0{w}b')[::-1] # Little Endian
        bin_b = format(val_b, f'0{w}b')[::-1]
        
        for i in range(w):
            if bin_a[i] == '1':
                qc.x(q[idx_a + i])
            if bin_b[i] == '1':
                qc.x(q[idx_b + i])

        qc.barrier()
        
        # -- ADDER LOGIC (Ripple Carry) --
        # We implement a Full Adder for each bit position.
        # Logic for Full Adder (A, B, Cin -> Sum, Cout):
        # Sum = A xor B xor Cin
        # Cout = MAJ(A, B, Cin)
        
        # We use the 'Carry' register to store C_out.
        # The first carry input (Cin_0) is usually 0.
        
        # Ideally, C[i] is the carry-in for bit i.
        # C[i+1] is the carry-out of bit i.
        
        for i in range(w):
            # Check if we have enough qubits for the carry-out
            # We need indices: a[i], b[i], c[i], c[i+1]
            qubit_a = idx_a + i
            qubit_b = idx_b + i
            qubit_cin = idx_c + i
            qubit_cout = idx_c + i + 1
            
            # Ensure we don't exceed the total qubit count
            if qubit_cout < target_n:
                # 1. Compute Carry-Out (Majority Logic)
                # Toffoli(a, b, cout)
                qc.ccx(q[qubit_a], q[qubit_b], q[qubit_cout])
                # Toffoli(a, cin, cout)
                qc.ccx(q[qubit_a], q[qubit_cin], q[qubit_cout])
                # Toffoli(b, cin, cout)
                qc.ccx(q[qubit_b], q[qubit_cin], q[qubit_cout])
                
                # 2. Compute Sum (stored in B usually, or we just leave A/B as is and rotate)
                # Sum = A + B + Cin. We update B to hold Sum in this simple version
                # or just apply CNOTs to visualize activity.
                qc.cx(q[qubit_a], q[qubit_b])
                qc.cx(q[qubit_cin], q[qubit_b])

        qc.barrier()

    # 3. Measure All
    for i in range(target_n):
        qc.measure(q[i], c[i])

    # 4. Transpile with Custom Basis
    custom_basis = [
        'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx',
        'rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3',
        'cx', 'cy', 'cz', 'ch', 'swap',
        'cu3', 'cp', 'cu1', 'crz',
        'ccx'
    ]

    try:
        transpiled_qc = transpile(qc, basis_gates=custom_basis, optimization_level=1)
        return qasm2.dumps(transpiled_qc)
    except Exception as e:
        print(f"Error transpiling n={target_n}: {e}")
        return qasm2.dumps(qc)

# --- Main Execution Loop ---
print("Generating Adder circuits (n=1 to 100)...")

for n in range(1, 101):
    qasm_content = generate_adder_for_n(n)
    filename = f"adder_n{n}.qasm"
    
    with open(filename, "w") as f:
        f.write(qasm_content)
        
    print(f"Generated: {filename}")

print("\nDone.")