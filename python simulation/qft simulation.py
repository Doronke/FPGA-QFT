import numpy as np
from qiskit import QuantumCircuit, Aer, execute

def qft_circuit(n):
    circuit = QuantumCircuit(n)
    for j in range(n):
        for k in range(j):
            circuit.cp(2 * np.pi / 2**(j - k), k, j)  # Use cp instead of cu1
        circuit.h(j)
    return circuit

# Example usage:
n_qubits = 4
qft = qft_circuit(n_qubits)

# Simulate the QFT
simulator = Aer.get_backend('statevector_simulator')
result = execute(qft, simulator).result()
state_vector = result.get_statevector()

print("State vector after QFT:", state_vector)
