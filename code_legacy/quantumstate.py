import cmath
from collections import deque
from register import Qubit, QuantumRegister
import random
import numpy as np

class QuantumState:
    """
    Manages a queue of quantum registers.

    This class simulates a quantum circuit by applying gates to all registers
    in the queue and handles conceptual 'splits' for two-qubit gates.
    """

    def __init__(self, num_qubits):
        """
        Initializes the queue with a single quantum register.

        Args:
            num_qubits (int): The number of qubits in the initial register.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        self._num_qubits = num_qubits
        self._registers_queue = deque([QuantumRegister(num_qubits)])

    def hadamard(self, qubit_index):
        """
        Applies a single-qubit Hadamard gate to the specified qubit in all
        registers currently in the queue.

        Args:
            qubit_index (int): The index of the qubit to apply the gate to.
        """
        # Apply the gate to every register in the queue
        for register in self._registers_queue:
            try:
                register.hadamard(qubit_index)
            except IndexError as e:
                print(f"Error applying Hadamard to qubit {qubit_index}: {e}")

    def u3(self, qubit_index, theta=0, phi=0, lambda_=0):
        for register in self._registers_queue:
            try:
                register.u3(qubit_index, theta, phi, lambda_)
            except IndexError as e:
                print(f"Error applying Hadamard to qubit {qubit_index}: {e}")
                
    def cnot(self, qubit_index1, qubit_index2):
        """
        Simulates the application of a cnot gate.

        Args:
            qubit_index1 (int): The index of the first target qubit.
            qubit_index2 (int): The index of the second target qubit.
        
        Raises:
            ValueError: If the two qubit indices are the same.
        """
        if qubit_index1 == qubit_index2:
            raise ValueError("The two qubit indices must be different.")
        
        # get the current len of _registers_queue
        k = len(self._registers_queue)
        # Apply the gate to every register in the queue
        for i in range(k):
            register = self._registers_queue[i]
            new_register = register.cnot(qubit_index1, qubit_index2)
            if not new_register is None:
                self._registers_queue.append(new_register)

    def cu3(self, qubit_index1, qubit_index2, theta=0, phi=0, lambda_=0):
        """
        Simulates the application of a controlled u3 gate.

        Args:
            qubit_index1 (int): The index of the first target qubit.
            qubit_index2 (int): The index of the second target qubit.
        
        Raises:
            ValueError: If the two qubit indices are the same.
        """
        if qubit_index1 == qubit_index2:
            raise ValueError("The two qubit indices must be different.")
        
        # get the current len of _registers_queue
        k = len(self._registers_queue)
        # Apply the gate to every register in the queue
        for i in range(k):
            register = self._registers_queue[i]
            new_register = register.cu3(qubit_index1, qubit_index2, theta, phi, lambda_)
            if not new_register is None:
                self._registers_queue.append(new_register)
    

    def measure(self, shot=1<<10):
        result = {}
        for _ in range(shot):
            rand_num = random.uniform(0, 1)
        
            selected_register = None
            cumulative_weight = 0
            
            # TODO: check whether it is necessary to implement binary search
            # it is clear that now the time complexity is O(N), 
            # but binary search may cause O(2^E) memory complexity
            
            for register in self._registers_queue:
                cumulative_weight += abs(register._weight)**2
                if rand_num <= cumulative_weight:
                    selected_register = register
                    break
            measurement = selected_register.measure()
            if measurement not in result:
                result[measurement] = 0
            result[measurement] += 1
        return dict(sorted(result.items(), key=lambda x: x[0]))
    
    def statevector(self):
        # only used for validation
        # only work for small N so far because of copy
        statevector = np.zeros(2**self._num_qubits, dtype=complex)
        for register in self._registers_queue:
            statevector += register.statevector()

        return statevector
        

    def __str__(self):
        """
        Returns a string representation of all registers in the queue.
        """
        if not self._registers_queue:
            return "The quantum register queue is empty."
            
        queue_str = "Current Registers in Queue:\n"
        for i, register in enumerate(self._registers_queue):
            queue_str += f"\n--- Register {i + 1} ({register.num_qubits} qubits) ---\n"
            queue_str += str(register) + "\n"
            
        return queue_str


if __name__ == "__main__":
    print("--- Initializing a quantum register queue ---")
    registers = QuantumState(2)
    print(registers)
    
    print("\n==================================================================\n")
    print("--- Applying a Hadamard gate to Qubit 0 ---")
    registers.hadamard(0)
    print(registers)


    print("\n==================================================================\n")
    print("--- Applying a CNOT gate to Qubit 0, 1 ---")
    registers.cnot(0, 1)
    print(registers)

    print("\n==================================================================\n")
    print("--- Measure ---")
    print(registers.measure())