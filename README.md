# FPGA-QFT

In this project we aim to take the known QFT algorithm, 'translating' aim into classical reprsentation using a new method, as explained in more details in the book project and the following article: [Simulation of quantum algorithms using classical probabilistic bits and circuits](https://arxiv.org/abs/2307.14452) 

implementing both in python and on an FPGA device(Virtex VC709) while comparing the runtime of each implementation. for the python code we have the file: '2qbit_simulation.py' where we simulate the 2 q-bits case, and for the FPGA device we use 'SerialRead.py' to read the data sent from the FPGA to the pc, and 'matrix mutiplication' which consist all the needed modules to run the 2 q-bits case on the FPGA device, including the communication interface between the PC and the FPGA using UART-to-USB bridge provided on board.
we will explain each part in details.

additionaly a python script running the QFT algorithm using qiskit was used for comperison purpuses, named 'qskitQFT.py'.
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

# 4qbit_simulation.py - Quantum Gates and Sparse Matrix Operations

This script defines and manipulates quantum gates for a 4-qubit system, represents them using both dense and sparse matrices, and simulates a 4-qubit quantum algorithm.

## Key Components

### 1. Helper Functions
- **print_sparse_matrix_for_verilog(name, sparse_matrix, filename)**: Converts a sparse matrix to a Verilog-compatible format, including information about non-zero elements.
- **make_spars(matrix, tolerance=1e-8)**: Converts a given dense matrix to its sparse matrix representation using the Compressed Sparse Row (CSR) format.

### 2. Matrix Definitions and Quantum Gates
- **Zero Matrix**: `zero_2x2` is a 2x2 matrix filled with zeros.
- **Identity Matrices**: `I4`, `I8`, and `I2` are 4x4, 8x8, and 2x2 identity matrices, respectively.
- **Probabilistic Operations**: `P0` and `P1` are defined as probabilistic operations.
- **NOT Gate**: `NOT` is a 2x2 matrix representing the NOT gate.
- **Hadamard Gate**: `H` is a 2x2 Hadamard gate.
- **Phase Gates**: `MP_pi_2`, `MP_pi_4`, and `MP_pi_8` represent phase gates with different angles.

### 3. Composite Gates
- **Controlled-NOT (CNOT) Gates**: `M_CNOT14`, `M_CNOT23`, `M_CNOT41`, and `M_CNOT32` are defined for different qubit pairs.
- **SWAP Gates**: `M_SWAP14` and `M_SWAP23` are defined using sequences of CNOT gates.
- **Controlled-Phase Gates**: `MCP34`, `MCP24`, `MCP23`, `MCP14`, `MCP13`, and `MCP12` are implemented for various qubit combinations.

### 4. Quantum States
- **Single Qubit States**: `q0` and `q1` represent the |0⟩ and |1⟩ states.
- **Classical Base States**: `s0` and `s1` are defined as normalized classical base states.
- **Entangled States**: `s01`, `s10`, `s00`, and `s11` represent entangled states of 2 qubits.

### 5. Simulation
The simulation applies a sequence of gates to an initial state and measures the runtime for both dense and sparse matrix representations.

## Usage
1. Run the script to define all gates, states, and helper functions.
2. The simulation section applies various gates to an initial state and measures the performance for both dense and sparse matrices.
3. Use `make_spars()` to create sparse representations of matrices.

## Example
To create a sparse representation of the CNOT gate and print it in Verilog format:
```python
sM_CNOT14 = make_spars(M_CNOT14)
```

## Notes
- The code uses a custom fixed-point format with 24 bits for Verilog representation.
- The script is designed for a 4-qubit system but can be extended for larger systems.

# neqbit_gates.py - Multi-Controlled Gates

This script purpose is to help create n-qubits control gates. the CPHASE creator still requires more work.

## Key Components

### 1. Helper Functions
- **print_sparse_matrix_for_verilog(name, sparse_matrix, filename)**: Same as in the 4qbit_simulation.py script.
- **make_spars(matrix, tolerance=1e-8)**: Enhanced version that handles both numpy arrays and scipy sparse matrices.

### 2. Multi-Controlled Gate Functions
- **make_mcgate_1n(n, P0, P1, gate)**: Creates a multi-controlled gate with control on the first qubit and target on the n-th qubit.
- **make_mcgate_n1(n, P0, P1, gate)**: Creates a multi-controlled gate with control on the n-th qubit and target on the first qubit.
- **make_mcgate_jk(n, j, k, P0, P1, gate)**: Creates a multi-controlled gate with control on the j-th qubit and target on the k-th qubit.
- **make_mcpgate_jk(n, j, k, P0, P1, gate)**: Creates a multi-controlled phase gate.

### 3. SWAP Gate Functions
- **make_swapmcgate_1n(n, P0, P1)**: Creates a multi-controlled SWAP gate between the first and n-th qubit.
- **make_swapmcgate_n1(n, P0, P1)**: Creates a multi-controlled SWAP gate between the n-th and first qubit.
- **make_swapmcgate_jk(n, j, k, P0, P1)**: Creates a multi-controlled SWAP gate between the j-th and k-th qubit.

### 4. Utility Functions
- **from_Q_to_C(gate)**: Converts a quantum gate to its classical representation.
- **make_MP(phi)**: Creates a phase gate with the given angle.

## Usage
1. Use the multi-controlled gate functions to create complex gates for n-qubit systems.
2. Apply these gates in quantum circuit simulations.
3. Use `make_spars()` to create sparse representations of the gates.

## Example
To create a multi-controlled phase gate and its sparse representation:
```python
MCSWAP23 = make_swapmcgate_jk(4,2,3, P0, P1) 
make_spars(MCSWAP23)
```

## Notes
- This script is more flexible and can handle n-qubit systems.
- It provides a framework for creating complex multi-controlled gates efficiently using sparse matrix operations.

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

# QFT Implementation using Qiskit 'qiskitQFT.py'

This script implements the Quantum Fourier Transform (QFT) algorithm using Qiskit, IBM's open-source framework for quantum computing. It includes functionality to create QFT circuits, simulate them, and estimate their execution time on real quantum hardware.

## Features

- Implements QFT for a variable number of qubits
- Simulates the QFT circuit using Qiskit's QASM simulator
- Measures simulation time on a classical computer
- Estimates execution time on a real quantum computer
- Provides performance metrics for QFT circuits with 2 to 10 qubits

## Requirements

- Python 3.7 or higher
- Qiskit

To install Qiskit, run:

```
pip install qiskit
```

## Usage

1. Save the script as `qiskitQFT.py` (or any preferred name).
2. Run the script using Python:

```
python qiskitQFT.py
```

## Output Explanation

The script will output:

1. The QFT circuit for 3 qubits (as an example).
2. Simulation time for the 3-qubit circuit on a classical computer.
3. Estimated execution time for the 3-qubit circuit on a real quantum computer.
4. Performance metrics (simulation time and estimated execution time) for QFT circuits with 2 to 10 qubits.

### Understanding the Output

- **Simulation Time**: This is the actual time taken to simulate the circuit on your classical computer using Qiskit's QASM simulator.
- **Estimated Execution Time**: This is a rough approximation of how long the circuit might take to run on a real quantum computer. It's based on typical gate times (50 ns for single-qubit gates, 300 ns for two-qubit gates).

Note: The estimated execution time is a theoretical approximation and may differ significantly from actual run times on real quantum hardware due to various factors like qubit connectivity, hardware-specific gate times, and error rates.

## Customization

You can modify the script to:

- Change the number of qubits in the example circuit (default is 3).
- Adjust the range of qubits in the performance metrics loop (default is 2 to 10).
- Modify the estimated gate times for more accurate hardware-specific estimates.

## Further Reading

To learn more about the Quantum Fourier Transform and Qiskit, check out:

- [Qiskit Textbook - Quantum Fourier Transform](https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html)
- [Qiskit Documentation](https://qiskit.org/documentation/)


# 
# FGPA modules

The FPGA consist of 2 main modules, one for the calculation of the QFT classical form, and one for the communication between the PC and the FPGA using the UART-USB bridge.
the QFT module is a sraight foward module to run the algorithm using sparse matrices, while the the communication module is built out of 2 modules consisted of different state machin, 'state_wm.sv' file consist of the state machine to 'speak' with the spga, 'axi_uart_wait_full_tx.v' checks to see if the UART buffer status and accordinly 'state_wm.sv' continue or stop sending info from the FPGA device.  'matrix_nwm.sv' connect mention before modules so they will work together as intended.

each module explained with more details:

# AXI UART Wait Full TX Module - 'axi_uart_wait_full_tx.v'

This SystemVerilog module, `axi_uart_wait_full_tx`, is designed for reading data from an AXI UART interface and determining whether the UART transmission buffer is full. The module operates through a series of states to manage the AXI read transactions and provides a signal (`continue1`) to indicate when the transmission buffer is not full.

## Module Interface

### Inputs
- **clk**: Clock signal.
- **activate**: Main activation signal for the module.

### Outputs
- **continue1**: Signal to indicate when the UART transmission buffer is not full.
- **arvalid**: Signal to indicate the validity of the read address.
- **araddr**: Address for the read transaction.
- **rready**: Signal to indicate readiness to receive read data.

### Inputs (from AXI)
- **arready**: Signal from AXI indicating readiness to accept the read address.
- **rvalid**: Signal from AXI indicating valid read data.
- **rdata**: Read data from the AXI interface.

### Outputs (to AXI)
- **rresp**: Read response from the AXI interface.

## Parameters
- **AXI_ADDR_WIDTH**: Width of the AXI address (set to 4).
- **AXI_DATA_WIDTH**: Width of the AXI data (set to 32).

## State Machine

The module uses a finite state machine (FSM) with the following states:
1. **INIT**: Initial state, waiting to start the read transaction.
2. **SET_ADDRESS**: Set the read address to the UART status register.
3. **SET_VALID**: Assert the read address valid signal.
4. **AWAIT_READY**: Wait for the AXI interface to acknowledge the read address.
5. **AWAIT_VALID_DATA**: Wait for the valid read data signal from the AXI interface.
6. **READ_DATA**: Read the data from the AXI interface and check if the transmission buffer is full.
7. **SET_READ_READY**: Assert the read data ready signal.
8. **HALT**: Determine the next action based on the transmission buffer status.

## Operation

- When the module is activated (`activate` is high), it begins the process of reading the UART status register.
- The read address (`araddr`) is set to `0x08` and the read transaction is initiated by asserting `arvalid`.
- The module waits for `arready` from the AXI interface, then waits for `rvalid` to receive the read data (`rdata`).
- The module checks if the transmission buffer is full (`tx_full`) by inspecting the appropriate bit of `rdata`.
- If the buffer is full, the module resets to the initial state. If not, it sets the `continue1` signal high and halts.

## Usage

1. **Instantiation**: Instantiate the module in your top-level design and connect the appropriate signals.
2. **Activation**: Drive the `activate` signal high to start the read process.
3. **Output**: Monitor the `continue1` signal to check if the UART transmission buffer is not full.

## Notes

- The module assumes the UART status register is located at address `0x08`. Adjust `araddr` as necessary for your specific design.
- The `tx_full` signal is derived from the 4th bit of the read data (`rdata[3]`). Adjust this bit position according to your UART status register definition.


## `state_wm.sv` Module

### Overview
The `state_wm` module is a SystemVerilog module designed for managing the state transitions required for data transfer via an AXI UART Lite interface. The module handles data input, state transitions, and interfacing with the AXI UART Lite and an additional FIFO wait module.

### Module Description

#### Ports
- **Inputs:**
  - `clk`: Clock signal.
  - `s_axi_aresetn`: Asynchronous reset signal.
  - `rx`: Receive data from UART.
  - `data`: 8-bit data input.
  - `valid`: Data validity flag.

- **Outputs:**
  - `tx`: Transmit data to UART.
  - `led0` to `led6`: LED outputs for status indication.
  - `ready`: Data ready signal.

#### Internal Signals and Registers
- **AXI Interface:**
  - `s_axi_awaddr`, `s_axi_awvalid`, `s_axi_awready`: AXI write address channel signals.
  - `s_axi_wdata`, `s_axi_wstrb`, `s_axi_wvalid`, `s_axi_wready`: AXI write data channel signals.
  - `s_axi_bready`, `s_axi_bvalid`, `s_axi_bresp`: AXI write response channel signals.
  - `s_axi_araddr`, `s_axi_arvalid`, `s_axi_arready`: AXI read address channel signals.
  - `s_axi_rready`, `s_axi_rvalid`, `s_axi_rresp`, `s_axi_rdata`: AXI read data channel signals.

- **Other Signals:**
  - `interrupt`: Interrupt signal from UART.
  - `s_axi_aclk`: Clock signal for AXI interface.
  - `wait_fifo_active`: Flag indicating if FIFO wait is active.
  - `fifo_not_full`: Signal indicating if the FIFO is not full.
  - `data_in`: Register to hold input data.
  - `s_axi_awaddr_as`, `s_axi_wdata_as`: Intermediate signals for AXI write address and data.

#### State Machine
The module uses a state machine with the following states:
- `INIT`: Initial state, waits for valid data.
- `SET_ADD`: Sets the AXI write address.
- `SET_DATA`: Sets the AXI write data.
- `SET_VALID`: Sets the AXI valid signals.
- `AWAIT_READY`: Waits for AXI interface to be ready.
- `WAIT_FOR_TX_FIFO`: Waits for the TX FIFO to be not full.
- `HALT`: Halts the state machine and sets the ready signal.

### Dependencies
- `axi_uartlite_0`: Instance of the AXI UART Lite module.
- `axi_uart_wait_full_tx`: Instance of a module that manages waiting for the TX FIFO to be not full.

### Usage
This module is designed to interface with an AXI UART Lite module and manage data transfer via a state machine. The state transitions are controlled based on the `valid` input signal and the status of the AXI interface signals.

### Additional Comments
This module is a part of a larger project involving serial communication and requires an AXI UART Lite IP core for proper functioning. Ensure that the AXI UART Lite IP core is correctly instantiated and connected in your design. The module also includes additional LED outputs for debugging and status indication purposes.

# Matrix NWM Module - 'matrix_nwm.sv' 

This SystemVerilog module, `matrix_nwm`, is designed to interface with an AXI UART and manage data transmission based on the status of a matrix. It operates through a finite state machine (FSM) to control the flow of data and state transitions.

## Module Interface

### Inputs
- **clk**: Clock signal.
- **rx**: UART receive signal.
- **s_axi_aresetn**: Active low reset signal for the AXI interface.

### Outputs
- **tx**: UART transmit signal.
- **led1**: LED output to indicate status.

## Internal Signals and Registers

### Wires
- **s01_out [63:0]**: 24-bit signed output data array from the matrix module.
- **counter**: 24-bit counter value.
- **overflow**: Overflow signal from the matrix module.
- **dataw**: Wire to hold data.
- **data2w**: Wire to hold additional data.

### Registers
- **i**: 2-bit register to manage state transitions within `SET_DATA`.
- **data**: 8-bit register to hold data to be transmitted.
- **pos_counter**: 8-bit register to keep track of the current position.
- **pos_counter2**: 8-bit register to buffer position updates.
- **over_flow**: 24-bit register to hold overflow data.
- **valid**: Register to indicate valid data transmission.
- **data2**: 8-bit register to hold additional data for transmission.
- **data3 [1:0]**: Array of 24-bit registers to hold multiple data values.
- **led1**: Register to control the LED output.
- **current_state**: 4-bit register to hold the current state of the FSM.
- **next_state**: 4-bit register to hold the next state of the FSM.

### Initial Values
- **i = 2'b00**
- **valid = 0**
- **pos_counter = 8'h0**
- **data = 8'h00**
- **data2 = 8'h00**
- **current_state = INIT**

## State Machine

The module operates through a finite state machine (FSM) with the following states:
1. **INIT**: Initial state, waiting for reset and matrix readiness.
2. **SET_DATA**: Set data from the matrix to be transmitted.
3. **UPDATE_POS**: Update the position counter.
4. **POS_BUFFER**: Buffer the updated position and data.
5. **SEND_DATA**: Send the data when the UART is ready.
6. **WAIT_4_READY**: Wait for the UART to be ready for the next data.
7. **HALT**: Halt state, waiting for reset to reinitialize.

## Dependencies
- **state_wm**: Submodule handling state management for the AXI UART interface.
- **twobytwo_sv_timing**: Submodule handling matrix operations and providing output data.

## Operation

1. **Initialization**: Upon reset (`s_axi_aresetn`), the module initializes its internal registers and enters the `INIT` state.
2. **Matrix Data Handling**: In the `INIT` state, if the reset is deasserted and the matrix is ready, the module captures overflow and counter data from the matrix.
3. **Data Setting and Transmission**: The module transitions through `SET_DATA`, `UPDATE_POS`, `POS_BUFFER`, and `SEND_DATA` states to prepare and send data from the matrix.
4. **Waiting for UART Readiness**: In the `WAIT_4_READY` state, the module waits for the UART to be ready for the next data transmission.
5. **Halting**: If the reset signal is asserted, the module transitions to the `HALT` state and waits for the reset to be deasserted to reinitialize.

## Usage

1. **Instantiation**: Instantiate the module in your top-level design and connect the appropriate signals.
2. **Activation**: Drive the `s_axi_aresetn` signal low to reset the module and high to start operation.
3. **Monitoring**: Monitor the `tx` signal for UART transmission and `led1` for status indication.

## Notes

- The module interfaces with submodules `state_wm` and `twobytwo_sv_timing` for handling AXI UART communication and matrix operations, respectively.
- Ensure that the reset signal (`s_axi_aresetn`) is properly controlled to initialize and operate the module
- the LEDs are not used in the final code, they can be used for debugging purposes
- they are 2 SET_DATA states one sending the final result vector from running the QFT algorithm, while the other is sending the total clock cycles needed for the calculation/overflow data. it's possible to switch between the two by taking the states in and out of commenting. data3[0] holds overflow bit, while data3[1] holds clock cycles counter.


# 'twobytwo_sv_timing.sv' Module

## Overview

This SystemVerilog module, `twobytwo_sv_timing`, is designed to transform quantum bits (qubits) into classical bits and apply a series of quantum gates including the Hadamard gate, the Controlled-Phase (CPHASE) gate, and Controlled-NOT (CNOT) gates. This transformation and gate application are part of a Quantum Fourier Transform (QFT) algorithm.

## Module Description

The `twobytwo_sv_timing` module takes two qubit inputs (`q0_in` and `q1_in`) and a clock signal (`clk`). It outputs several classical bit sequences and signals, including:

- `s0_out`
- `s1_out`
- `s01_out`
- `prob00`
- `prob01`
- `prob10`
- `prob11`
- `counter`
- `overflow`
- `matrix_ready`

The module implements the following functionality:

1. **Qubit to Classical Transformation**: Converts qubit inputs into classical representations.
2. **Hadamard Gate**: Applies the Hadamard gate to the qubits, represented as sparse matrices.
3. **CPHASE Gate**: Applies the Controlled-Phase gate to the qubits.
4. **CNOT Gates**: Applies Controlled-NOT gates, specifically `CNOT12` and `CNOT21`, as part of a SWAP operation.

## Port Definitions

- `input q0_in`: Input for the first qubit.
- `input q1_in`: Input for the second qubit.
- `input clk`: Clock signal.
- `output [23:0] s0_out [7:0]`: Output representing the transformed state of qubit 0.
- `output [23:0] s1_out [7:0]`: Output representing the transformed state of qubit 1.
- `output [23:0] s01_out [63:0]`: Output representing the combined state of qubits 0 and 1.
- `output [23:0] prob00`: Probability amplitude for the |00⟩ state.
- `output [23:0] prob01`: Probability amplitude for the |01⟩ state.
- `output [23:0] prob10`: Probability amplitude for the |10⟩ state.
- `output [23:0] prob11`: Probability amplitude for the |11⟩ state.
- `output [23:0] counter`: Counter for iterations.
- `output overflow`: Overflow signal.
- `output matrix_ready`: Indicates when the matrix operations are ready.

## Internal Registers and Parameters

The module uses several internal registers and parameters to hold intermediate values and control states:

- `reg signed [23:0] s0_out [7:0]`: Registers for `s0_out`.
- `reg signed [23:0] s1_out [7:0]`: Registers for `s1_out`.
- `reg signed [23:0] s01_out [63:0]`: Registers for `s01_out`.
- `reg signed [47:0] s0_temp [7:0]`: Temporary registers for `s0_out`.
- `reg signed [47:0] s1_temp [7:0]`: Temporary registers for `s1_out`.
- `reg signed [47:0] s01_temp [63:0]`: Temporary registers for `s01_out`.
- `reg signed [23:0] prob00`, `prob01`, `prob10`, `prob11`: Probability amplitudes.
- `reg signed [47:0] prob00_temp`, `prob01_temp`, `prob10_temp`, `prob11_temp`: Temporary probability amplitudes.
- `reg signed [23:0] counter`: Counter register.
- `reg signed overflow`: Overflow register.
- `integer i`, `j`, `k`, `start`: Loop and control variables.
- `reg [3:0] current_state`, `next_state`: State machine registers.
- `typedef enum {INIT, HADAMARD, TEMP_MOVE, JOIN01, CP, TEMP_01, CN12, TEMP_12, CN21, TEMP_21, CN12_2, TEMP_f, HALT} states`: State enumeration.

## State Machine

The module uses a state machine to control the sequence of operations:

1. **INIT**: Initializes the module and checks if the qubit inputs have changed.
2. **HADAMARD**: Applies the Hadamard gate using temporary registers.
3. **TEMP_MOVE**: Moves the Hadamard results back to the output registers.
4. **JOIN01**: Combines the transformed states of `s0_out` and `s1_out`.
5. **CP**: Applies the CPHASE gate using temporary registers.
6. **TEMP_01**: Moves the CPHASE results back to the output registers.
7. **CN12**: Applies the first CNOT12 gate.
8. **TEMP_12**: Moves the results of the first CNOT12 gate back to the output registers.
9. **CN21**: Applies the CNOT21 gate.
10. **TEMP_21**: Moves the results of the CNOT21 gate back to the output registers.
11. **CN12_2**: Applies the second CNOT12 gate.
12. **TEMP_f**: Final temporary state to clean up any remaining operations.
13. **HALT**: Halts the state machine when operations are complete.

## Additional Notes

- Ensure that the clock signal `clk` is properly generated and connected to the module.
- Monitor the `matrix_ready` signal to determine when the matrix operations are complete and the output values are valid.
- The state machine controls the sequence of operations and ensures that each quantum gate is applied in the correct order.
- The following code looke at the 2 q-bits case alone.


# Documentation and Resources

### User Guide for VC709 Evaluation Board and FPGA
- [VC709 Evaluation Board User Guide](https://www.mouser.com/datasheet/2/903/ug887-vc709-eval-board-v7-fpga-1596461.pdf)

### LogiCore IP Guide for UART-USB Bridge
- [AXI UART Lite LogiCore IP Guide (AMD)](https://docs.amd.com/v/u/en-US/axi_uartlite_ds741)
- [AXI UART Lite LogiCore IP Guide (Hthreads)](https://hthreads.github.io/modules/eecs-4114/data-sheets/pg142-axi-uartlite.pdf)

### GitHub Repository for AXI UART Demo Project
- [AXI UART Demo Project](https://github.com/DouglasWWolf/axi_uart_demo/blob/main/xilinx_zcu104/axi_uart_zcu104.gen/sources_1/bd/design_1/design_1.bxml)

### AXI Reference Guide
- [AXI Reference Guide](https://www.xilinx.com/support/documents/ip_documentation/axi_ref_guide/latest/ug761_axi_reference_guide.pdf)

### Theoretical background
- [Simulation of quantum algorithms using classical probabilistic bits and circuits](https://arxiv.org/abs/2307.14452) 


