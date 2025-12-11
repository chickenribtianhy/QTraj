import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import qasm2  # Required for Qiskit 1.0+ logic

def generate_adder_qasm(n_bits):
    """
    Generates OpenQASM 2.0 code for a Ripple Carry Adder (Cuccaro)
    with 'n_bits' size, preserving CCX gates.
    """
    
    # 1. Define Registers
    # cin: Carry-in bit (helper for the first addition)
    q_cin = QuantumRegister(1, 'cin')
    # a: Input number A (also acts as the carry chain)
    q_a = QuantumRegister(n_bits, 'a')
    # b: Input number B (stores the result/SUM at the end)
    q_b = QuantumRegister(n_bits, 'b')
    # cout: Carry-out bit (stores the overflow/MSB)
    q_cout = QuantumRegister(1, 'cout')
    # ans: Classical register to store the final measurement
    c_ans = ClassicalRegister(n_bits + 1, 'ans')

    qc = QuantumCircuit(q_cin, q_a, q_b, q_cout, c_ans)

    # --- HELPER FUNCTIONS ---
    # Majority Gate: Logic for Carry generation
    def majority(circuit, c, b, a):
        circuit.cx(a, b)
        circuit.cx(a, c)
        circuit.ccx(c, b, a)

    # UnMajority and Add: Logic for Sum generation and Carry uncomputation
    def unmajority_add(circuit, c, b, a):
        circuit.ccx(c, b, a)
        circuit.cx(a, c)
        circuit.cx(a, b)

    # --- BUILD ADDER LOGIC ---
    
    # 2. Forward Pass (Calculate Carries)
    # Ripple from Least Significant Bit (LSB) to Most Significant Bit (MSB)
    if n_bits > 0:
        majority(qc, q_cin[0], q_b[0], q_a[0])
        
    for i in range(1, n_bits):
        majority(qc, q_a[i-1], q_b[i], q_a[i])

    # 3. Connect to Final Carry Out
    # The last carry from a[n-1] goes to cout
    if n_bits > 0:
        qc.cx(q_a[n_bits-1], q_cout[0])

    # 4. Backward Pass (Calculate Sum and Restore A)
    # Reverse the process to compute Sum bits in 'b' and clean up 'a'
    for i in range(n_bits - 1, 0, -1):
        unmajority_add(qc, q_a[i-1], q_b[i], q_a[i])
        
    if n_bits > 0:
        unmajority_add(qc, q_cin[0], q_b[0], q_a[0])

    # 5. Measure All
    # We measure 'b' (which now holds the Sum) and 'cout' (the overflow)
    for i in range(n_bits):
        qc.measure(q_b[i], c_ans[i])
    qc.measure(q_cout[0], c_ans[n_bits])

    # --- EXPORT TO QASM ---
    return qasm2.dumps(qc)

def main():
    # Create a directory to store the files
    output_dir = "/home/htian02/qsim/cuda_impl/benchmarks/qasm/adder_2/"
    # /home/htian02/qsim/cuda_impl/benchmarks/qasm/adder_2/adder_generate_qasm.py
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #     print(f"Created directory: {output_dir}")

    print("Generating files from n=1 to n=35...")

    # Loop from 1 to 35
    for n in range(1, 36):
        # Generate QASM content
        qasm_content = generate_adder_qasm(n)
        
        # Define filename (e.g., adder_n4.qasm)
        filename = f"adder_n{n}.qasm"
        filepath = os.path.join(output_dir, filename)
        
        # Write to file
        with open(filepath, "w") as f:
            f.write(qasm_content)
        
        print(f"Generated: {filepath}")

    print("\nDone! All 35 files have been generated.")

if __name__ == "__main__":
    main()