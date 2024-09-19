import numpy as np
from scipy import sparse
import inspect
import time

# =============================================================================
# 
# =============================================================================

def print_sparse_matrix_for_verilog(name, sparse_matrix, filename):
    with open(filename, 'w') as file:
        file.write(f"\n// Sparse matrix {name} \n")
        rows, cols = sparse_matrix.shape
        nnz = sparse_matrix.nnz
        
        file.write(f"\nlocalparam int {name}_rows = {rows};\n")
        file.write(f"localparam int {name}_cols = {cols};\n")
        file.write(f"localparam int {name}_nnz = {nnz};\n\n")
        
        # Write row indices
        file.write(f"localparam [{int(np.ceil(np.log2(rows))-1)}:0] row{name} [{nnz-1}:0] = {{\n")
        row_indices = sparse_matrix.nonzero()[0]
        file.write(', '.join(map(str, row_indices)))
        file.write("};\n\n")
        
        # Write column indices
        file.write(f"localparam [{int(np.ceil(np.log2(cols))-1)}:0] column{name} [{nnz-1}:0] = {{\n")
        col_indices = sparse_matrix.nonzero()[1]
        file.write(', '.join(map(str, col_indices)))
        file.write("};\n\n")
        
        # Write values
        file.write(f"localparam [23:0] value{name} [{nnz-1}:0] = {{\n")
        values = sparse_matrix.data
        
        def format_value(v):
            # Convert to custom fixed-point format
            is_negative = v < 0
            abs_v = abs(v)
            whole_part = int(abs_v)
            frac_part = abs_v - whole_part
            
            # Ensure whole part fits in 4 bits
            if whole_part > 7 and not is_negative:
                raise ValueError(f"Positive whole part of {v} exceeds 4-bit capacity")
            if whole_part > 8 and is_negative:
                raise ValueError(f"Negative whole part of {v} exceeds 4-bit capacity")
            
            # Convert fractional part to binary (19 bits)
            frac_binary = 0
            for i in range(19):
                frac_part *= 2
                if frac_part >= 1:
                    frac_binary |= (1 << (18 - i))
                    frac_part -= 1
            
            # Combine whole and fractional parts
            value = (whole_part << 19) | frac_binary
            
            # Apply custom two's complement for negative numbers
            if is_negative:
                value = (~value + 1) & 0xFFFFFF  # Flip bits and add 1
            
            # Format the binary string with an underscore after the first 5 bits
            binary_str = f"{value:024b}"
            formatted_str = binary_str[:5] + '_' + binary_str[5:]
            
            return f"24'b{formatted_str}"
        
        formatted_values = [format_value(v) for v in values]
        file.write(', '.join(formatted_values))
        file.write("};\n")
        file.write("\n")



def make_spars(matrix, tolerance=1e-8):
    # Convert to sparse matrix with small numbers treated as 0
    a = sparse.csr_matrix(matrix, dtype=np.float32)
    
    # Apply tolerance
    a.data[abs(a.data) < tolerance] = 0
    a.eliminate_zeros()
    
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    b = [matrix_name for matrix_name, matrix_val in callers_local_vars if matrix_val is matrix]
    a.maxprint = np.inf
    filename = f"{b[0]}.txt"
    print_sparse_matrix_for_verilog(b[0], a, filename)
    return a

# =============================================================================
# ##################################### BASIC GATES ############################################    
# =============================================================================

start_time = time.time()

# Define 2x2 zero matrix
zero_2x2 = np.zeros((2, 2))
# Probabilistic operations P0 and P1
P0 = np.array([[1, 0], [0, 0]]) 
P1 = np.array([[0, 0], [0, 1]])
I4 = np.eye(4)
I8 = np.eye(8)
I2 = np.eye(2)

sP0 = make_spars(P0)
sP1 = make_spars(P1)
sI4 = make_spars(I4)
sI8 = make_spars(I8)

P0_tensor = np.kron(I4, P0)
P1_tensor = np.kron(I4, P1)

sP0_tensor = make_spars(P0_tensor)
sP1_tensor = make_spars(P1_tensor)

# =============================================================================
# ##################################### END BASIC GATES ############################################    
# =============================================================================


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


#Controlled-NOT for 4 qbit

M_I2 = np.kron(I4 , I2)
M_I4 = np.kron(M_I2 , M_I2)
M_I8 = np.kron(M_I2 , np.kron(M_I2,M_I2))

M_CNOT14 = np.kron(P0_tensor, M_I8) + np.kron(P1_tensor, np.kron(M_I4,M_NOT))
M_CNOT23 = np.kron(np.kron(M_I2, (np.kron(P0_tensor,M_I2)+np.kron(P1_tensor,M_NOT))),M_I2)
M_CNOT41 = np.kron(M_I8 , P0_tensor) + np.kron(M_I4 ,np.kron(M_NOT,P1_tensor))
M_CNOT32 = np.kron(np.kron(M_I2, (np.kron(M_I2,P0_tensor)+np.kron(M_NOT,P1_tensor))),M_I2)

sM_CNOT14 = make_spars(M_CNOT14)
sM_CNOT23 = make_spars(M_CNOT23)
sM_CNOT41 = make_spars(M_CNOT41)
sM_CNOT32 = make_spars(M_CNOT32)
# =============================================================================
# ########### END NOT GATE ############
# =============================================================================

# =============================================================================
# ########### SWAP GATE ############
# =============================================================================

M_SWAP14 = np.matmul(np.matmul(M_CNOT14 , M_CNOT41) , M_CNOT14)
M_SWAP23 = np.matmul(np.matmul(M_CNOT23 , M_CNOT32) , M_CNOT23)

sM_SWAP14 = make_spars(M_SWAP14)
sM_SWAP23 = make_spars(M_SWAP23)

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
# ########### END HADAMARD GATE ############
# =============================================================================

# =============================================================================
# ########### PHASE GATE ############
# =============================================================================

def make_MP(phi):
    # Imaginary part of P(phi)
    Im_P_phi = np.array([[0, 0], [0, np.sin(phi)]])
    # Real part of P(phi)
    Re_P_phi = np.array([[1, 0], [0, np.cos(phi)]])
    # Combined operation M[P(phi)]
    M_P_phi = np.block([[Re_P_phi, zero_2x2, zero_2x2, Im_P_phi],
                        [zero_2x2, Re_P_phi, Im_P_phi, zero_2x2],
                        [Im_P_phi, zero_2x2, Re_P_phi, zero_2x2],
                        [zero_2x2, Im_P_phi, zero_2x2, Re_P_phi]])
    return M_P_phi
    
    
    

MP_pi_2 = make_MP(np.pi/2)

sMP_pi_2 = make_spars(MP_pi_2)

MP_pi_4 = make_MP(np.pi/4)

sMP_pi_4 = make_spars(MP_pi_4)

MP_pi_8 = make_MP(np.pi/8)

sMP_pi_8 = make_spars(MP_pi_8)

MCP34 = np.kron(P0_tensor,M_I2) + np.kron(P1_tensor , MP_pi_2)
MCP24 = np.kron(P0_tensor,M_I4) + np.kron(P1_tensor , np.kron(M_I2, MP_pi_4))
MCP23 = np.kron(np.kron(P0_tensor , M_I2) + np.kron(P1_tensor, MP_pi_2) , M_I2)
MCP14 = np.kron(P0_tensor, M_I8) + np.kron(P1_tensor, np.kron(M_I4, MP_pi_8))
MCP13 = np.kron(np.kron(P0_tensor,M_I4) + np.kron(P1_tensor, np.kron(M_I2, MP_pi_4)), M_I2)
MCP12 = np.kron(np.kron(P0_tensor, M_I2) + np.kron(P1_tensor, MP_pi_2), M_I4) 

sMCP34 = make_spars(MCP34)
sMCP24 = make_spars(MCP24)
sMCP23 = make_spars(MCP23)
sMCP14 = make_spars(MCP14)
sMCP13 = make_spars(MCP13)
sMCP12 = make_spars(MCP12)

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
s0 = np.array([2,1,0,1,1,1,1,1])
s1 = np.array([1,2,1,0,1,1,1,1])

ss0 = make_spars(s0)
ss1 = make_spars(s1)

u = np.array([1,1,1,1,1,1,1,1])
p0 = s0 - u
p1 = s1 - u

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
# ########### 4qbit algorithm simulation ############
# =============================================================================

start_simulation1_time = time.time()

Hs0 = np.matmul(M_H , p0) 
Hs1 = np.matmul(M_H , p1)

Hs00 = (np.kron(u,u) +np.kron(Hs0,Hs0))
STEP_CP34_S00 = np.matmul(MCP34, Hs00) + (np.matmul((np.eye(64) - MCP34), np.kron(u,u))) 

Hs000 = np.kron(u,np.kron(u,u))+np.kron(Hs0,STEP_CP34_S00 - np.kron(u,u))
STEP_CP24_S000 = np.matmul(MCP24 , Hs000) + (np.matmul((np.eye(512) - MCP24) ,  np.kron(u,np.kron(u,u))))
STEP_CP23_S000 = np.matmul(MCP23 , STEP_CP24_S000) +  (np.matmul((np.eye(512) - MCP23) ,  np.kron(u,np.kron(u,u))))

Hs0000 = np.kron(np.kron(u,u),np.kron(u,u))+np.kron(Hs0,STEP_CP23_S000 - np.kron(u,np.kron(u,u)))
STEP_CP14_S0000 = np.matmul(MCP14 , Hs0000) + (np.matmul((np.eye(4096) - MCP14) , np.kron(np.kron(u,u),np.kron(u,u))))
STEP_CP13_S0000 = np.matmul(MCP13 , STEP_CP14_S0000) + (np.matmul((np.eye(4096) - MCP13) , np.kron(np.kron(u,u),np.kron(u,u))))
STEP_CP12_S0000 = np.matmul(MCP12 , STEP_CP13_S0000) + (np.matmul((np.eye(4096) - MCP12) , np.kron(np.kron(u,u),np.kron(u,u))))

STEP_SWAP14_S0000 = np.matmul(M_SWAP14, STEP_CP12_S0000) + ((np.matmul((np.eye(4096) - M_SWAP14) , np.kron(np.kron(u,u),np.kron(u,u)))))
STEP_SWAP23_S0000 = np.matmul(M_SWAP23, STEP_SWAP14_S0000) + ((np.matmul((np.eye(4096) - M_SWAP23) , np.kron(np.kron(u,u),np.kron(u,u)))))

prob0000 = (1-STEP_SWAP23_S0000[0])**2 + (1-STEP_SWAP23_S0000[4])**2

end_time = time.time()
non_sparse_runtime = end_time - start_time
non_sparse_runtime_sim_time = start_simulation1_time - end_time




end_time2 = time.time()
sparse_runtime = end_time2 - start_time - non_sparse_runtime_sim_time

print(f"Non-sparse simulation runtime: {non_sparse_runtime:.6f} seconds")
print(f"Sparse simulation runtime: {sparse_runtime:.6f} seconds")



# =============================================================================
# # Example usage
# sample_matrix = sparse.csr_matrix([[1, 0, 0, 2.5], 
#                                    [0, -3.25, 0, 0], 
#                                    [0, 0, 6.875, 0], 
#                                    [3.25, 0, 0, -6.875]])
# # Write the sparse matrix to a text file
# print_sparse_matrix_for_verilog("example", sample_matrix, "example.txt")
# =============================================================================
