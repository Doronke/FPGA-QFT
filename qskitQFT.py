from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram
import numpy as np
import time

def qft(n):
    """Implement the Quantum Fourier Transform for n qubits."""
    qr = QuantumRegister(n)
    circuit = QuantumCircuit(qr)
    
    def swap_registers(circuit, n):
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        return circuit
    
    def qft_rotations(circuit, n):
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(np.pi/2**(n-qubit), qubit, n)
        qft_rotations(circuit, n)
    
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def simulate_and_time_qft(n, shots=1):
    """
    Create a QFT circuit for n qubits, simulate it, and measure execution time.
    
    Args:
    n (int): Number of qubits
    shots (int): Number of shots for the simulation
    
    Returns:
    tuple: (QuantumCircuit, float, float) - The QFT circuit, simulation time, and estimated execution time
    """
    start_time = time.time()
    qft_circuit = qft(n)
    
    # Add measurement to all qubits
    qft_circuit.measure_all()
    
    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    

    result = execute(qft_circuit, simulator, shots=shots).result()
    end_time = time.time()
    simulation_time = end_time - start_time
    
    # Estimate execution time on a real quantum computer
    # This is a rough estimate based on typical gate times
    single_qubit_gate_time = 50e-9  # 50 nanoseconds
    two_qubit_gate_time = 300e-9  # 300 nanoseconds
    
    total_single_qubit_gates = qft_circuit.count_ops()['h'] + qft_circuit.count_ops()['swap']
    total_two_qubit_gates = qft_circuit.count_ops()['cp']
    
    estimated_execution_time = (total_single_qubit_gates * single_qubit_gate_time +
                                total_two_qubit_gates * two_qubit_gate_time)
    
    return qft_circuit, simulation_time, estimated_execution_time

# Example usage
n = 40  # Number of qubits
qft_circuit, simulation_time, estimated_execution_time = simulate_and_time_qft(n)
print(f"QFT Circuit for {n} qubits:")
print(qft_circuit)
print(f"\nSimulation time: {simulation_time:.6f} seconds")
print(f"Estimated execution time on a real quantum computer: {estimated_execution_time:.9f} seconds")

# Measure simulation and estimated execution time for different numbers of qubits
#print("\nPerformance for different numbers of qubits:")
#for n in range(2, 5):
 #   _, simulation_time, estimated_execution_time = simulate_and_time_qft(n)
   # print(f"{n} qubits - Simulation: {simulation_time:.6f} s, Estimated execution: {estimated_execution_time:.9f} s")