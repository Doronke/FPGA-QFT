# FPGA-QFT

In this project we aim to take the known QFT algorithm, 'translating' aim into classical reprsentation using a new method, as explained in more details in the book project and the following article: [Simulation of quantum algorithms using classical probabilistic bits and circuits](https://arxiv.org/abs/2307.14452) 

implementing both in python and on an FPGA device(Virtex VC709) while comparing the runtime of each implementation. for the python code we have the file: '2qbit_simulation.py' where we simulate the 2 q-bits case, and for the FPGA device we use 'SerialRead.py' to read the data sent from the FPGA to the pc, and 'matrix mutiplication' which consist all the needed modules to run the 2 q-bits case on the FPGA device, including the communication interface between the PC and the FPGA using UART-to-USB bridge provided on board.
we will explain each part in details.

# 2qbit_simulation.py - Quantum Gates and Sparse Matrix Operations

This code defines and manipulates various quantum gates, represents them using both dense and sparse matrices,translating them to the classical presentation used as shown in the book project and the article, and simulates a 2-qubit quantum algorithm. The simulation includes gates like NOT, SWAP, Hadamard, and Phase gates..

## Key Components

### 1. Helper Functions
- **print_matrix(name, matrix)**: Prints a matrix with real and imaginary parts formatted to three decimal places.
- **make_spars(matrix)**: Converts a given dense matrix to its sparse matrix representation using the Compressed Sparse Row (CSR) format.
- **print_matrix_for_verilog(name, matrix)**: Converts a matrix to a format suitable for Verilog hardware description language.
- **print_sparse_matrix_for_verilog(name, sparse_matrix)**: Converts a sparse matrix to a Verilog-compatible format, including information about non-zero elements.

### 2. Matrix Definitions and Quantum Gates
- **Zero Matrix**: `zero_2x2` is a 2x2 matrix filled with zeros.
- **Identity Matrices**: `I4` and `I8` are 4x4 and 8x8 identity matrices, respectively.
- **Probabilistic Operations**: `P0` and `P1` are defined as probabilistic operations.
- **NOT Gate**: `NOT` is a 2x2 matrix representing the NOT gate.
- **Hadamard Gate**: `H` is a 2x2 Hadamard gate.
- **Phase Gate**: `P(phi)` represents the phase gate with a phase angle `phi`.

### 3. Composite Gates
- **Controlled-NOT (CNOT) Gates**: `M_CNOT12` and `M_CNOT21` are defined using tensor products and probabilistic operations.
- **SWAP Gate**: `M_SWAP` is defined using a sequence of CNOT gates.
- **Controlled-Phase Gate**: `M_CP_phi` combines probabilistic operations and phase gates.

### 4. Quantum States
- **Single Qubit States**: `q0` and `q1` represent the |0⟩ and |1⟩ states.
- **Classical Base States**: `s0` and `s1` are defined as normalized classical base states.
- **Entangled States**: `s01`, `s10`, `s00`, and `s11` represent entangled states of 2 qubits.

### 5. Simulation
The simulation applies a sequence of gates to an initial state and measures the runtime for both dense and sparse matrix representations.
- **Gate Applications**: Hadamard (`M_H`), Controlled-Phase (`M_CP_phi`), Controlled-NOT (`M_CNOT12` and `M_CNOT21`), and SWAP (`M_SWAP`) gates are applied to the state.
- **Runtime Measurement**: The runtime for the simulation using dense matrices (`non_sparse_runtime`) and sparse matrices (`sparse_runtime`) is calculated and printed.

### Usage
To use this code:
1. Run the script to define all gates, states, and helper functions.
2. To print a matrix in a readable format, use the `print_matrix` function.
3. To convert a matrix to Verilog format, use the `print_matrix_for_verilog` or `print_sparse_matrix_for_verilog` functions.
4. The simulation section applies various gates to an initial state and measures the performance for both dense and sparse matrices.
5. in the make_spars(matrix) it's possible to print the sparse matrix informations taking down the comments in the function.

### Example
To print the SWAP gate matrix in Verilog format:
```python
print_matrix_for_verilog("M_SWAP", M_SWAP)
```
To print the sparse representation of the SWAP gate in Verilog format:
```python
print_sparse_matrix_for_verilog("sM_SWAP", sM_SWAP)
```

### Notes
- The phase angle `phi` is set to π/2 but can be changed as needed.
- The Verilog conversion assumes a fixed-point representation with 24 bits.
- The code is designed for both educational and practical purposes, providing insights into quantum gate operations and their efficient representation.
- the code is tailored for the 2 q-bits case only.



# SerialRead.py - Serial Data Processing Script

This Python script reads data from a serial port, processes the data into 24-bit words, and prints each word in both binary and hexadecimal formats. The script is useful for reading and analyzing data from devices connected via a serial port.
the 24-bit broken down to 3-1 byte words format was chosen as it fits the length of each word calculated in the FPGA device, while being able to send 1 byte word at a time from the UART-USB bridge.

## Key Components

### 1. Function Definitions
- **process_data(data)**:
  - Takes a list of bytes as input.
  - Processes the data into 24-bit words.
  - Prints each 24-bit word in binary and hexadecimal formats.
  - Returns a list of 24-bit words.

### 2. Serial Port Configuration
- **serial.Serial()**: Initializes the serial port.
- **ser.port**: Specifies the COM port (e.g., 'COM3').
- **ser.baudrate**: Sets the baud rate (e.g., 115200).

### 3. Main Loop
- Opens the serial port and continuously reads incoming data.
- When data is available, it reads and prints the data in hexadecimal format.
- Processes the data into 24-bit words using the `process_data` function.
- Handles keyboard interrupt to exit the program gracefully.

## Usage

### Setup
1. Ensure you have the `pyserial` library installed. You can install it using:
   ```sh
   pip install pyserial
   ```

### Configuration
- Replace `'COM3'` with your actual COM port.
- Adjust the `baudrate` to match your device's settings.

### Exiting the Script
- Press `Ctrl+C` to exit the script.

### Notes
- The script assumes data is received in multiples of 3 bytes. If the data length is not a multiple of 3, the remaining bytes are ignored.
- The script reads and prints all incoming data in hexadecimal format before processing it into 24-bit words.
- Modify the script as needed to match your specific requirements and device settings.

### Links
For more information on:
- [pySerial Documentation](https://pyserial.readthedocs.io/)
- [Serial Communication](https://en.wikipedia.org/wiki/Serial_communication)
