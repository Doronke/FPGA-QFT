import numpy as np
from scipy import sparse
import inspect

def print_matrix(name, matrix):
    print(f"{name}:")
    matrix = np.atleast_2d(matrix)  # Ensure the input is at least 2D
    rows, cols = matrix.shape
    max_real_length = max(len(f"{abs(num.real):.3f}") for num in matrix.flatten())
    max_imag_length = max(len(f"{abs(num.imag):.3f}") for num in matrix.flatten())
    
    for i in range(rows):
        print("[", end=" ")
        for j in range(cols):
            num = matrix[i, j]
            if np.isclose(num.imag, 0):
                print(f"{num.real:>{max_real_length}.3f}", end="  ")
            elif np.isclose(num.real, 0):
                print(f"{num.imag:>{max_imag_length}.3f}j", end="  ")
            else:
                print(f"{num.real:>{max_real_length}.3f}{num.imag:>{max_imag_length+1}.3f}j", end="  ")
        print("]")
    print()
    
def make_spars(matrix):
    a = sparse.csr_matrix(matrix)
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    b =  [matrix_name for matrix_name, matrix_val in callers_local_vars if matrix_val is matrix]
    a.maxprint = np.inf
    print(b , ":\n" ,a)
    return a
    

# =============================================================================
# ##################################### GATES ############################################    
# =============================================================================
# Define 2x2 zero matrix
zero_2x2 = np.zeros((2, 2))
# Probabilistic operations P0 and P1
P0 = np.array([[1, 0], [0, 0]]) 
P1 = np.array([[0, 0], [0, 1]])
I4 = np.eye(4)
I8 = np.eye(8)

sP0 = make_spars(P0)
sP1 = make_spars(P1)
sI4 = make_spars(I4)
sI8 = make_spars(I8)


P0_tensor = np.kron(I4, P0)
P1_tensor = np.kron(I4, P1)

sP0_tensor = make_spars(P0_tensor)
sP1_tensor = make_spars(P1_tensor)


# =============================================================================
# ########### NOT GATE ############
# =============================================================================

# NOT gate
NOT = np.array([[0, 1],
                [1, 0]])

sNOT = make_spars(NOT)

# Matrix representation of NOT gate with 2x2 zero matrices
M_NOT = np.block([[NOT, zero_2x2, zero_2x2, zero_2x2],
                  [zero_2x2, NOT, zero_2x2, zero_2x2],
                  [zero_2x2, zero_2x2, NOT, zero_2x2],
                  [zero_2x2, zero_2x2, zero_2x2, NOT]])

sM_NOT = make_spars(M_NOT)


# Controlled-NOT12 gate using tensor product
M_CNOT12 = np.kron(P0_tensor, I8) + np.kron(P1_tensor, M_NOT)
M_CNOT21 = np.kron(I8 , P0_tensor) + np.kron(M_NOT , P1_tensor)

sM_CNOT12 = make_spars(M_CNOT12)
sM_CNOT21 = make_spars(M_CNOT21)

# =============================================================================
# ########### END NOT GATE ############
# =============================================================================

# =============================================================================
# ########### SWAP GATE ############
# =============================================================================

M_SWAP = np.matmul(np.matmul(M_CNOT12 , M_CNOT21) , M_CNOT12)

sM_SWAP = make_spars(M_SWAP)

# =============================================================================
# ########### END SWAP GATE ############
# =============================================================================

# =============================================================================
# ########### HADMARD GATE ############
# =============================================================================

H = np.array([[1/np.sqrt(2) , 1/np.sqrt(2)] , [1/np.sqrt(2) , -1/np.sqrt(2)]])

sH = make_spars(H)

M_H = np.block([[H, zero_2x2, zero_2x2, zero_2x2],
                  [zero_2x2, H, zero_2x2, zero_2x2],
                  [zero_2x2, zero_2x2, H, zero_2x2],
                  [zero_2x2, zero_2x2, zero_2x2, H]])

sM_H = make_spars(M_H)

# =============================================================================
# ########### END HADAMARED GATE ############
# =============================================================================

# =============================================================================
# ########### PHASE GATE ############
# =============================================================================

# Phase operation P(phi)
phi = np.pi / 2  # You can change this value as needed

# Imaginary part of P(phi)
Im_P_phi = np.array([[0, 0], [0, np.sin(phi)]])

sIm_P_phi = make_spars(Im_P_phi)

# Real part of P(phi)
Re_P_phi = np.array([[1, 0], [0, np.cos(phi)]])

sRe_P_phi = make_spars(Re_P_phi)

# Combined operation M[P(phi)]
M_P_phi = np.block([[Re_P_phi, zero_2x2, zero_2x2, Im_P_phi],
                    [zero_2x2, Re_P_phi, Im_P_phi, zero_2x2],
                    [Im_P_phi, zero_2x2, Re_P_phi, zero_2x2],
                    [zero_2x2, Im_P_phi, zero_2x2, Re_P_phi]])

sM_P_phi = make_spars(M_P_phi)

# Controlled-phase operation M[CP(phi)] using tensor product
M_CP_phi = np.kron(P0_tensor, I8) + np.kron(P1_tensor, M_P_phi)

sM_CP_phi = make_spars(M_CP_phi)

# =============================================================================
# ########### END PHASE GATE ############
# =============================================================================

# =============================================================================
# ########### END GATES ############
# =============================================================================

# =============================================================================
# ########### STATE VECTOR ############
# =============================================================================

#quantum starting state for a single bit
q0 = np.array([1,0])
q1 = np.array([0,1])

sq0 = make_spars(q0)
sq1 = make_spars(q1)

#classical base states
s0 = 1/8*np.array([2,1,0,1,1,1,1,1])
s1 = 1/8*np.array([1,2,1,0,1,1,1,1])

ss0 = make_spars(s0)
ss1 = make_spars(s1)

u = np.array([1,1,1,1,1,1,1,1])
p0 = 8*s0 - u
p1 = 8*s1 - u

su = make_spars(u)
sp0 = make_spars(p0)
sp1 = make_spars(p1)

#entangled state of 2 qbits
s01 = (1/(8**2))*(np.kron(u,u) +np.kron(p0,p1))
s10 = (1/(8**2))*(np.kron(u,u) +np.kron(p1,p0))
s00 = (1/(8**2))*(np.kron(u,u) +np.kron(p0,p0))
s11 = (1/(8**2))*(np.kron(u,u) +np.kron(p1,p1))

ss01 = make_spars(s01)
ss10 = make_spars(s10)
ss00 = make_spars(s00)
ss11 = make_spars(s11)

# =============================================================================
# ########### END STATE VECTOR ############
# =============================================================================

# =============================================================================
# ########### 2qbit algorithm simulation ############
# =============================================================================

Hs0 = np.matmul(M_H , p0) 
Hs1 = np.matmul(M_H , p1)
Hs01_entangled = np.kron(u, u) + np.kron(Hs0 , Hs1)
Hs01_CP = np.matmul(M_CP_phi , Hs01_entangled) + np.matmul((np.eye(64) - M_CP_phi) , np.kron(u,u))
Hs01_CNOT12 = np.matmul(M_CNOT12 , Hs01_CP) + np.matmul((np.eye(64) - M_CNOT12) , np.kron(u,u))
Hs01_CNOT21 = np.matmul(M_CNOT21,Hs01_CNOT12) + np.matmul((np.eye(64) - M_CNOT21) , np.kron(u,u))
Hs01_CNOT12_2 = np.matmul(M_CNOT12 , Hs01_CNOT21) + np.matmul((np.eye(64) - M_CNOT12) , np.kron(u,u))
Hs01_SWAP = np.matmul(M_SWAP , Hs01_CP) + np.matmul((np.eye(64) - M_SWAP) , np.kron(u,u))

sHs0 = make_spars(Hs0)
sHs1 = make_spars(Hs1)
sHs01_entangled = make_spars(Hs01_entangled)
sHs01_CP = make_spars(Hs01_CP)
sHs01_CNOT12 = make_spars(Hs01_CNOT12)
sHs01_CNOT21 = make_spars(Hs01_CNOT21)
sHs01_CNOT12_2 = make_spars(Hs01_CNOT12_2)
sHs01_SWAP = make_spars(Hs01_SWAP)

# =============================================================================
# ########### 2qbit algorithm simulation ############
# =============================================================================


# Print the matrices using the custom function
# =============================================================================
# print_matrix("NOT12", NOT)
# print_matrix("M[NOT12]", M_NOT)
# print_matrix("M[CNOT12]", M_CNOT12)
# print_matrix("M[P(phi)]", M_P_phi)
# print_matrix("M[CP(phi)]", M_CP_phi)
# =============================================================================

def print_matrix_for_verilog(name, matrix):
    matrix = np.atleast_2d(matrix)  # Ensure the input is at least 2D
    rows, cols = matrix.shape
    print(f"{name}[{rows*cols-1}:0] = {{")
    
    for i in range(rows):
        line = ""
        for j in range(cols):
            num = matrix[i, j]
            if np.isclose(num.imag, 0):
                binary = format(int(num.real * (2**23)) & 0xFFFFFF, '024b')
            elif np.isclose(num.real, 0):
                binary = format(int(num.imag * (2**23)) & 0xFFFFFF, '024b')
            else:
                real_part = int(num.real * (2**22)) & 0x7FFFFF
                imag_part = int(num.imag * (2**22)) & 0x7FFFFF
                binary = format((real_part << 1) | (imag_part & 1), '024b')
            
            line += f"24'b{binary}, "
        
        print(line)
    
    print("};")
    print()

def print_sparse_matrix_for_verilog(name, sparse_matrix):
    print(f"// Sparse matrix {name} (classical representation)")
    rows, cols = sparse_matrix.shape
    nnz = sparse_matrix.nnz
    
    print(f"localparam int {name}_rows = {rows};")
    print(f"localparam int {name}_cols = {cols};")
    print(f"localparam int {name}_nnz = {nnz};")
    print(f"typedef struct packed {{")
    print(f"    logic [15:0] row;")
    print(f"    logic [15:0] col;")
    print(f"    logic signed [23:0] value;")
    print(f"}} sparse_element;")
    print(f"localparam sparse_element {name}[{nnz-1}:0] = {{")
    
    non_zero = sparse_matrix.nonzero()
    data = sparse_matrix.data
    
    elements = []
    for i, j, v in zip(non_zero[0], non_zero[1], data):
        binary_value = format(int(v * (2**23)) & 0xFFFFFF, '024b')
        element = f"{{16'd{i}, 16'd{j}, 24'b{binary_value}}}"
        elements.append(element)
    
    # Join elements with commas and newlines
    formatted_elements = ',\n'.join(elements)
    
    print(formatted_elements)
    print("};")
    print()

# Example usage:
print_sparse_matrix_for_verilog("sM_SWAP", sM_SWAP)

# Example usage:
print_matrix_for_verilog("M_SWAP", M_SWAP)

# Example usage:
#print_matrix_for_verilog("CSWAP", M_SWAP)