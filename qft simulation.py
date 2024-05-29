from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import numpy as np

def qft(circ, q):
    """Apply the quantum Fourier transform to the qubits in circ"""
    n = len(q)
    # Apply the Hadamard gates and controlled-phase gates
    for j in range(n):
        circ.h(q[j])
        for k in range(j+1, n):
            angle = 2 * np.pi / (2**(k-j+1))
            circ.cp(angle, q[k], q[j])
    # Permute the qubits
    for i in range(n // 2):
        circ.swap(q[i], q[n - i - 1])
    return circ

# Define the number of qubits
n = 4

# Create a quantum circuit
qc = QuantumCircuit(n, n)

# Apply the forward QFT
qft(qc, range(n))

# Measure the qubits
qc.measure(range(n), range(n))

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
qobj = assemble(compiled_circuit)
result = simulator.run(qobj).result()

# Plot the results
counts = result.get_counts(qc)
plot_histogram(counts)