import cmath
import math
import random
import numpy as np

class Qubit:
    """
    Represents a single qubit with its state defined by two complex numbers,
    alpha and beta.
    
    The state of the qubit is a superposition of the |0> and |1> basis states:
    
    |psi> = alpha|0> + beta|1>
    
    The condition |alpha|^2 + |beta|^2 = 1 must always hold.
    """
    
    def __init__(self, alpha=1.0, beta=0.0):
        """
        Initializes a Qubit object.
        
        Args:
            alpha (complex): The complex amplitude for the |0> state.
            beta (complex): The complex amplitude for the |1> state.
            
        Raises:
            ValueError: If the state is not a valid quantum state (i.e.,
                        if |alpha|^2 + |beta|^2 is not approximately equal to 1).
        """
        # Ensure the state is normalized. Use a small epsilon for floating-point comparison.
        if abs(alpha*alpha.conjugate() + beta*beta.conjugate() - 1.0) > 1e-9:
            raise ValueError("The sum of the squared magnitudes of alpha and beta must be equal to 1.")
            
        self._alpha = alpha
        self._beta = beta

    @property
    def alpha(self):
        """
        Returns the complex amplitude of the |0> state.
        """
        return self._alpha

    @property
    def beta(self):
        """
        Returns the complex amplitude of the |1> state.
        """
        return self._beta

        
    def __str__(self):
        """
        Returns a string representation of the qubit's state.
        """
        # Format the complex numbers for a clean string representation.
        alpha_str = f"({self.alpha.real:.4f}{'+' if self.alpha.imag >= 0 else ''}{self.alpha.imag:.4f}j)"
        beta_str = f"({self.beta.real:.4f}{'+' if self.beta.imag >= 0 else ''}{self.beta.imag:.4f}j)"
        return f"{alpha_str}|0⟩ + {beta_str}|1⟩"

    def _apply_hadamard(self):
        """
        Applies the Hadamard gate's transformation.
        This is an internal method to be called by the QuantumRegister.
        """
        new_alpha = (self.alpha + self.beta) / cmath.sqrt(2)
        new_beta = (self.alpha - self.beta) / cmath.sqrt(2)
        
        self._alpha = new_alpha
        self._beta = new_beta
    
    def _apply_not(self):
        """
        Applies the Pauli-X (NOT) gate's transformation.
        This is an internal method to be called by the QuantumRegister.
        """
        self._alpha, self._beta = self._beta, self._alpha
        
    def _apply_u3(self, theta, phi, lambda_):
        """
        Applies the U3 gate's transformation with parameters theta, phi, and lambda.
        This is an internal method to be called by the QuantumRegister.
        """
        # Current state vector [alpha, beta]
        alpha_old = self._alpha
        beta_old = self._beta
        
        # Pre-calculate trigonometric values and complex exponentials
        cos_theta_half = math.cos(theta / 2)
        sin_theta_half = math.sin(theta / 2)
        
        exp_lambda = cmath.exp(1j * lambda_)
        exp_phi = cmath.exp(1j * phi)
        
        # Apply the U3 matrix transformation:
        # | new_alpha | = | cos(t/2)            -e^(i*l)sin(t/2)    | | alpha_old |
        # | new_beta  |   | e^(i*p)sin(t/2)     e^(i*(p+l))cos(t/2) | | beta_old  |
        
        new_alpha = cos_theta_half * alpha_old - exp_lambda * sin_theta_half * beta_old
        new_beta = exp_phi * sin_theta_half * alpha_old + exp_phi * exp_lambda * cos_theta_half * beta_old
        
        self._alpha = new_alpha
        self._beta = new_beta


    def measure(self):
        """
        Converts the qubit state to a classical bit (0 or 1).
        """
        rand_num = random.uniform(0, 1)
        if rand_num < abs(self.alpha)**2:
            return '0'
        else:
            return '1'

class QuantumRegister:
    """
    Represents a quantum register consisting of N qubits.
    """
    
    def __init__(self, num_qubits, weight=1):
        """
        Initializes a quantum register with N qubits, all in the |0> state.
        
        Args:
            num_qubits (int): The number of qubits in the register.
            
        Raises:
            ValueError: If num_qubits is not a positive integer.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
            
        self._qubits = [Qubit() for _ in range(num_qubits)]
        self._weight = weight
        self._num_qubits = num_qubits

    def copy(self):
        """
        Creates and returns a new, independent copy of the current QuantumRegister instance.

        Returns:
            QuantumRegister: A new, independent copy of the register.
        """
        new_register = QuantumRegister(self.num_qubits, weight=self._weight)
        for i, qubit in enumerate(self._qubits):
            new_register._qubits[i] = Qubit(alpha=qubit.alpha, beta=qubit.beta)
        return new_register
    
    @property
    def num_qubits(self):
        """
        Returns the number of qubits in the register.
        """
        return len(self._qubits)
        
    def get_qubit(self, index):
        """
        Returns the Qubit object at a specific index.
        
        Args:
            index (int): The index of the qubit to retrieve.
            
        Returns:
            Qubit: The Qubit object at the specified index.
        
        Raises:
            IndexError: If the index is out of bounds.
        """
        if not 0 <= index < self.num_qubits:
            raise IndexError("Qubit index out of range.")
        return self._qubits[index]
        
    def __str__(self):
        """
        Returns a string representation of the entire quantum register's state.
        """
        states = [str(qubit) for qubit in self._qubits]
        return f"Weight: {self._weight}\n" + \
                "\n".join([f"Qubit {i}: {state}" for i, state in enumerate(states)])

    def hadamard(self, qubit_index):
        """
        Applies the Hadamard gate to a specific qubit in the register.
        
        Args:
            qubit_index (int): The index of the qubit to apply the gate to.
            
        Raises:
            IndexError: If the qubit_index is out of bounds.
        """
        if not 0 <= qubit_index < self.num_qubits:
            raise IndexError("Qubit index out of range.")
        
        target_qubit = self._qubits[qubit_index]
        target_qubit._apply_hadamard()
    
    def u3(self, qubit_index, theta=0, phi=0, lambda_=0):
        if not 0 <= qubit_index < self.num_qubits:
            raise IndexError("Qubit index out of range.")
        
        target_qubit = self._qubits[qubit_index]
        target_qubit._apply_u3(theta, phi, lambda_)

    def not_gate(self, qubit_index):
        """
        Applies the Pauli-X (NOT) gate to a specific qubit in the register.
        
        Args:
            qubit_index (int): The index of the qubit to apply the gate to.
            
        Raises:
            IndexError: If the qubit_index is out of bounds.
        """
        if not 0 <= qubit_index < self.num_qubits:
            raise IndexError("Qubit index out of range.")
            
        target_qubit = self._qubits[qubit_index]
        target_qubit._apply_not()

    def cnot(self, qubit_index1, qubit_index2):
        if not (0 <= qubit_index1 < self.num_qubits and 0 <= qubit_index2 < self.num_qubits):
            raise IndexError("Qubit index out of range.")
        if qubit_index1 == qubit_index2:
            raise ValueError("The two qubit indices must be different.")

        control_qubit = self._qubits[qubit_index1]
        target_qubit = self._qubits[qubit_index2]

        weight1 = control_qubit.alpha
        weight2 = control_qubit.beta

        # control qubit is |0>
        if abs(weight1) - 1.0 == 0:
            return None
        # control qubit is |1>
        if abs(weight2) - 1.0 == 0:
            self._qubits[qubit_index2]._apply_not()
            return None
        # Not doesn't make difference
        # TODO: check should it be alpha==beta or alpha^2==beta^2
        if target_qubit.alpha - target_qubit.beta == 0:
            return None

        temp_weight = self._weight
        # modify on this register: weight=weight1, control qubit set to |0>
        self._weight = weight1 * temp_weight
        self._qubits[qubit_index1] = Qubit(alpha=1, beta=0)

        # initialize a new register: weight=weight2, control qubit set to |1>, target qubit apply not
        new_register = self.copy()
        new_register._weight = weight2 * temp_weight
        new_register._qubits[qubit_index1] = Qubit(alpha=0, beta=1)
        new_register._qubits[qubit_index2]._apply_not()

        return new_register

    def cu3(self, qubit_index1, qubit_index2, theta=0, phi=0, lambda_=0):
        if not (0 <= qubit_index1 < self.num_qubits and 0 <= qubit_index2 < self.num_qubits):
            raise IndexError("Qubit index out of range.")
        if qubit_index1 == qubit_index2:
            raise ValueError("The two qubit indices must be different.")

        control_qubit = self._qubits[qubit_index1]
        target_qubit = self._qubits[qubit_index2]

        weight1 = control_qubit.alpha
        weight2 = control_qubit.beta

        # control qubit is |0>
        if abs(weight1) - 1.0 == 0:
            return None
        # control qubit is |1>
        if abs(weight2) - 1.0 == 0:
            self._qubits[qubit_index2]._apply_u3(theta, phi, lambda_)
            return None
        
        temp_weight = self._weight
        # modify on this register: weight=weight1, control qubit set to |0>
        self._weight = weight1 * temp_weight
        self._qubits[qubit_index1] = Qubit(alpha=1, beta=0)

        # initialize a new register: weight=weight2, control qubit set to |1>, target qubit apply u3
        new_register = self.copy()
        new_register._weight = weight2 * temp_weight
        new_register._qubits[qubit_index1] = Qubit(alpha=0, beta=1)
        new_register._qubits[qubit_index2]._apply_u3(theta, phi, lambda_)

        return new_register


    def measure(self):
        """
        Measures the quantum register, collapsing the superposition into a single
        classical bit string.
        
        Returns:
            str: The measured classical bit string.
        """
        measurement = ""
        for qubit in self._qubits:
            measurement += qubit.measure()
        return measurement
    
    def statevector(self):
        _statevector = np.zeros(2**self._num_qubits, dtype=complex)
        for i in range(2**self._num_qubits):
            k = i
            acc = self._weight
            for qubit in self._qubits[::-1]:
                index = k % 2
                k //= 2
                if index == 0:
                    acc *= qubit._alpha
                else:
                    acc *= qubit.beta
                if acc == 0:
                    break
            _statevector[i] = acc

        return _statevector





if __name__ == "__main__":
    # test qubit init
    from utils import generate_random_qubit_state
    random_alpha, random_beta = generate_random_qubit_state()
    random_qubit = Qubit(alpha=random_alpha, beta=random_beta)
    print(f"Qubit in a random state: {random_qubit}")
    
    print("\n==================================================================\n")

    # test quantum register init
    try:
        # Create a quantum register with 3 qubits
        my_register = QuantumRegister(3)
        print("Initial state of the 3-qubit register:")
        print(my_register)
        print("Accessing a specific qubit:")
        q_1 = my_register.get_qubit(1)
        print(f"State of Qubit 1: {q_1}")
    except ValueError as e:
        print(f"Error: {e}")
    print("\n==================================================================\n")
    # test Hadamard gate to a single qubit
    # Create a quantum register with 3 qubits
    my_register = QuantumRegister(3)
    print("Initial state of the 3-qubit register:")
    print(my_register)
    
    print("\n--- Applying Hadamard to Qubit 0 ---")
    my_register.hadamard(0)
    print(my_register)

    print("\n==================================================================\n")
    # test not gate
    print("--- Applying NOT gate to Qubit 1 ---")
    my_register.not_gate(1)
    print(my_register)

    print("\n==================================================================\n")
    # test cnot
    print("--- Applying CNOT(0, 1) ---")
    print("Before:")
    print(my_register)
    print("After:")
    new_register = my_register.cnot(0, 1)
    print(my_register)
    print(new_register)

    print("\n==================================================================\n")
    # test measure
    print("--- Measure ---")
    print(my_register.measure())