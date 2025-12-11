import os
import sys
import time
import subprocess
import pandas as pd
import qiskit.qasm2
from qiskit_aer import AerSimulator

# =================CONFIGURATION=================
BENCHMARK_DIR = "benchmarks"
OUTPUT_FILE = "benchmark_results.xlsx"
TARGET_PREFIX = "dnn" 
METHODS = [
        #   'statevector', 
            'matrix_product_state',
            # 'tensor_network'
            ]
DEFAULT_SHOTS = 1 << 20  # Fallback for non-QFT tasks
TIMEOUT_SECONDS = 60
# ===============================================

# --- WORKER FUNCTION (Runs in a separate process) ---
def worker_run_one(file_path, method, shots):
    """
    Now accepts 'shots' as an argument.
    """
    try:
        # Convert shots to int because sys.argv passes strings
        shots = int(shots)
        
        qc = qiskit.qasm2.load(file_path)
        sim = AerSimulator(method=method, device="GPU")
        
        start_time = time.time()
        job = sim.run(qc, shots=shots)
        result = job.result()
        end_time = time.time()
        
        print(f"{end_time - start_time:.6f}")
        sys.exit(0) 
        
    except Exception as e:
        print(f"Worker Error: {e}", file=sys.stderr)
        sys.exit(1)

# --- MANAGER FUNCTION (Main loop) ---
def run_benchmark_safe(file_path, method, shots):
    # Pass 'shots' to the worker subprocess
    cmd = [sys.executable, __file__, "--worker", file_path, method, str(shots)]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=TIMEOUT_SECONDS
        )
        
        if result.returncode != 0:
            err_msg = result.stderr.strip()
            if not err_msg: 
                err_msg = f"Crash/Segfault (Code {result.returncode})"
            print(f"    [Failed] {err_msg}")
            return None

        return float(result.stdout.strip())

    except subprocess.TimeoutExpired:
        print(f"    [Timeout] > {TIMEOUT_SECONDS}s")
        return "pass"
    except Exception as e:
        print(f"    [Subprocess Error] {e}")
        return None

def main():
    results = []
    
    if not os.path.exists(BENCHMARK_DIR):
        print(f"Error: Directory '{BENCHMARK_DIR}' not found.")
        return
        
    files = sorted([f for f in os.listdir(BENCHMARK_DIR) 
                    if f.startswith(TARGET_PREFIX) and f.endswith(".qasm")])
    
    print(f"Found {len(files)} benchmarks for '{TARGET_PREFIX}'. Processing safely...")

    for filename in files:
        file_path = os.path.join(BENCHMARK_DIR, filename)
        
        try:
            n_qubits = int(filename.split('_n')[1].split('.')[0])
        except IndexError:
            n_qubits = -1 

        # === DYNAMIC SHOT CALCULATION ===
        if "qft" in TARGET_PREFIX.lower() or "dnn" in TARGET_PREFIX.lower():
            # If QFT, use 2^(n+5)
            # Ensure n_qubits is valid, otherwise fallback
            if n_qubits > 0:
                current_shots = 1 << (n_qubits + 5)
            else:
                current_shots = DEFAULT_SHOTS
        else:
            # Otherwise use default
            current_shots = DEFAULT_SHOTS
        # =================================

        print(f"Running {filename} (Shots={current_shots})...")
        
        row = {"Filename": filename, "Qubits": n_qubits}

        for method in METHODS:
            col_name = {
                'statevector': 'SV (s)',
                'matrix_product_state': 'MPS (s)',
                'tensor_network': 'TN (s)'
            }.get(method, method)

            # --- SKIP LOGIC ---
            if method == 'statevector' and n_qubits >= 33:
                print(f"  -> {method}... [Skipped: n={n_qubits} >= 33]")
                row[col_name] = "skipped"
                continue
            
            if method == 'tensor_network' and n_qubits >= 36:
                print(f"  -> {method}... [Skipped: n={n_qubits} >= 36]")
                row[col_name] = "skipped"
                continue

            if method == 'matrix_product_state' and n_qubits >= 18:
                print(f"  -> {method}... [Skipped: n={n_qubits} >= 12]")
                row[col_name] = "skipped"
                continue
            # ------------------

            print(f"  -> {method}...", end=" ", flush=True)
            
            # Pass calculated shots to the safe runner
            runtime = run_benchmark_safe(file_path, method, current_shots)
            
            if isinstance(runtime, float):
                row[col_name] = runtime
                print(f"{runtime:.4f}s")
            elif runtime == "pass":
                row[col_name] = "pass"
                print("pass")
            else:
                row[col_name] = "/"

        results.append(row)

    df = pd.DataFrame(results)
    cols = ["Filename", "Qubits"] + [c for c in df.columns if c not in ["Filename", "Qubits"]]
    df = df[cols].sort_values(by="Qubits")
    
    try:
        if os.path.exists(OUTPUT_FILE):
            with pd.ExcelWriter(OUTPUT_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                df.to_excel(writer, index=False, sheet_name=TARGET_PREFIX)
            print(f"\nSuccess! Appended sheet '{TARGET_PREFIX}' to {OUTPUT_FILE}")
        else:
            df.to_excel(OUTPUT_FILE, index=False, sheet_name=TARGET_PREFIX)
            print(f"\nSuccess! Created {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving Excel: {e}")
        df.to_csv(f"{TARGET_PREFIX}_backup.csv", index=False)
        print("Saved backup CSV.")

if __name__ == "__main__":
    # Check for worker flag
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # Args: [script.py, --worker, file_path, method, shots]
        worker_run_one(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()