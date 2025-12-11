import os
import re

BENCHMARK_DIR = "qasm/dnn"

def fix_qasm_gates():
    files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".qasm")]
    
    print(f"Repairing {len(files)} files in '{BENCHMARK_DIR}'...")
    
    for filename in files:
        filepath = os.path.join(BENCHMARK_DIR, filename)
        
        with open(filepath, "r") as f:
            content = f.read()
        
        # 1. Replace lowercase 'u(' with Capital 'U('
        # The regex looks for 'u' followed immediately by '('
        # This prevents replacing valid words like 'measure' or 'include'
        new_content = re.sub(r'\bu\(', 'U(', content)
        
        # 2. Ensure qelib1.inc is there (just in case)
        if 'include "qelib1.inc";' not in new_content:
            new_content = new_content.replace('OPENQASM 2.0;', 'OPENQASM 2.0;\ninclude "qelib1.inc";')

        with open(filepath, "w") as f:
            f.write(new_content)

    print("Repairs complete. Try running the benchmark now.")

if __name__ == "__main__":
    fix_qasm_gates()