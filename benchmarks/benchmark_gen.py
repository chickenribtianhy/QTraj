import os
import qiskit.qasm2 
import warnings
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    DraperQFTAdder, 
    CDKMRippleCarryAdder, 
    QFTGate, 
    RealAmplitudes  # <--- NEW: Standard QNN Ansatz
)

# Suppress internal deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

output_dir = "benchmarks"
os.makedirs(output_dir, exist_ok=True)

BASIS_GATES = ['u', 'cx', 'id'] 

print(f"Generating benchmarks in: {output_dir}")

# output_dir_ghz = "qasm/ghz"
# os.makedirs(output_dir_ghz, exist_ok=True)
# # ==========================================
# # 1. QFT and GHZ (Loop total qubits 2 to 40)
# # ==========================================
# for n in range(2, 100):
    
#     # --- A. QFT ---
#     qc_qft = QuantumCircuit(n)
#     qc_qft.append(QFTGate(n), range(n))
#     qc_qft = transpile(qc_qft, basis_gates=BASIS_GATES, optimization_level=0)
#     qc_qft.measure_all(inplace=True)
    
#     with open(os.path.join(output_dir, f"qft_n{n}.qasm"), "w") as f:
#         qiskit.qasm2.dump(qc_qft, f)

    # --- B. GHZ State ---
# for n in range(100, 150):

#     qc_ghz = QuantumCircuit(n)
#     qc_ghz.h(0)
#     for i in range(n - 1):
#         qc_ghz.cx(i, i+1)
        
#     qc_ghz = transpile(qc_ghz, basis_gates=BASIS_GATES, optimization_level=0)
#     qc_ghz.measure_all(inplace=True)
    
#     with open(os.path.join(output_dir_ghz, f"ghz_n{n}.qasm"), "w") as f:
#         qiskit.qasm2.dump(qc_ghz, f)

# # ==========================================
# # 2. Adders (Loop register size 1 to 20)
# # ==========================================
# output_dir_adder = "qasm/adder"
# os.makedirs(output_dir_adder, exist_ok=True)
# for n in range(1, 100):
    
#     # --- C. Draper QFT Adder ---
#     qc_draper = DraperQFTAdder(num_state_qubits=n, kind="fixed")
#     qc_draper = transpile(qc_draper, basis_gates=BASIS_GATES, optimization_level=0)
#     qc_draper.measure_all(inplace=True)
    
#     with open(os.path.join(output_dir, f"adder_draper_n{qc_draper.num_qubits}.qasm"), "w") as f:
#         qiskit.qasm2.dump(qc_draper, f)

    # --- D. Ripple Carry Adder ---

    # qc_ripple = CDKMRippleCarryAdder(num_state_qubits=n, kind="full")
    # qc_ripple = transpile(qc_ripple, basis_gates=BASIS_GATES, optimization_level=0)
    # qc_ripple.measure_all(inplace=True)
    
    # with open(os.path.join(output_dir_adder, f"adder_n{qc_ripple.num_qubits}.qasm"), "w") as f:
    #     qiskit.qasm2.dump(qc_ripple, f)
    
    # print(f"Generated Adders for register size {n}")

# ==========================================
# 3. DNN / QNN (Loop total qubits 2 to 40)
# ==========================================
# Using RealAmplitudes (Common ansatz for classification/VQE)

output_dir_dnn = "qasm/dnn"
os.makedirs(output_dir_dnn, exist_ok=True)
for n in range(2, 51):
    
    # --- E. RealAmplitudes (DNN proxy) ---
    # reps=3 means 3 "layers" of entanglement + rotation
    qc_dnn = RealAmplitudes(num_qubits=n, entanglement='full', reps=3)
    
    # RealAmplitudes has unbound parameters (theta). We must bind them to run QASM.
    # We bind them to random values or zeros to make it a concrete circuit.
    qc_dnn = qc_dnn.assign_parameters([0.5] * qc_dnn.num_parameters)
    
    qc_dnn = transpile(qc_dnn, basis_gates=BASIS_GATES, optimization_level=0)
    qc_dnn.measure_all(inplace=True)
    
    with open(os.path.join(output_dir_dnn, f"dnn_n{n}.qasm"), "w") as f:
        qiskit.qasm2.dump(qc_dnn, f)
        
    print(f"Generated DNN (RealAmplitudes) for {n} qubits")

print("Generation Complete.")