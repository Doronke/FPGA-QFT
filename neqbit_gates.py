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
    if isinstance(matrix, np.ndarray):
        # For numpy arrays, convert to sparse matrix
        a = sparse.csr_matrix(matrix, dtype=np.float32)
    elif sparse.issparse(matrix):
        # For sparse matrices, create a copy
        a = matrix.copy()
    else:
        raise TypeError("Input must be a numpy array or a scipy sparse matrix")
    
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
# ##################################### BASIC GATES ###########################
# =============================================================================

start_time = time.time()

# Define 2x2 zero matrix
zero_2x2 = np.zeros((2, 2))
# Probabilistic operations P0 and P1
P0 = (np.array([[1, 0], [0, 0]])) 
P1 = (np.array([[0, 0], [0, 1]]))
I4 = np.eye(4)
I8 = np.eye(8)
I2 = np.eye(2)


# =============================================================================
# ##################################### END BASIC GATES #######################
# =============================================================================

# =============================================================================
# ########### GATES ############
# =============================================================================

# NOT gate
NOT = np.array([[0, 1],
                [1, 0]])


def make_mcgate_1n(n , P0, P1, gate):
    M_I2 = sparse.csr_matrix(np.kron(np.eye(4) , np.eye(2)),dtype = np.float32)
    M_I1 = sparse.csr_matrix(np.eye(1),dtype = np.float32)
    I_n1 = M_I2
    I_n2 = M_I1
    for i in range (n-2):
        I_n1 = sparse.kron(I_n1 , M_I2)
    #for i in range(n-3):
        I_n2 = sparse.kron(I_n2,M_I2)
    mcgate = sparse.kron(from_Q_to_C(P0),I_n1)+sparse.kron(from_Q_to_C(P1),sparse.kron(I_n2,from_Q_to_C(gate)))
    return mcgate

def make_mcgate_n1(n , P0, P1, gate):
    M_I2 = sparse.csr_matrix(np.kron(np.eye(4) , np.eye(2)),dtype = np.float32)
    M_I1 = sparse.csr_matrix(np.eye(1),dtype = np.float32)
    I_n1 = M_I2
    I_n2 = M_I1
    for i in range (n-2):
        I_n1 = sparse.kron(I_n1 , M_I2)
    #for i in range(n-2):
        I_n2 = sparse.kron(I_n2,M_I2)
    mcgate = sparse.kron(I_n1,from_Q_to_C(P0))+sparse.kron(I_n2,sparse.kron(from_Q_to_C(gate),from_Q_to_C(P1)))
    return mcgate

def make_mcgate_jk(n,j,k,P0,P1,gate):
    M_I2 = sparse.csr_matrix(np.kron(np.eye(4) , np.eye(2)),dtype = np.float32)
    #M_I1 = sparse.csr_matrix(np.eye(1),dtype = np.float32)
    I_j = M_I2
    I_nk = M_I2
    for i in range (j-3):
        I_j = sparse.kron(I_j , M_I2)
    for i in range(n-k-2):
        I_nk = sparse.kron(I_nk,M_I2)
    if j<k:
        mcgate = sparse.kron(I_j, sparse.kron(make_mcgate_1n(k-j+1, P0, P1, gate),I_nk))
    else:
        mcgate = sparse.kron(I_j, sparse.kron(make_mcgate_n1(j-k+1, P0, P1, gate),I_nk))
    return mcgate;

def make_mcpgate_jk(n,j,k,P0,P1,gate):
    M_I2 = sparse.csr_matrix(np.kron(np.eye(4) , np.eye(2)),dtype = np.float32)
    M_I1 = sparse.csr_matrix(np.eye(1),dtype = np.float32)
    I_j = M_I1
    I_nk = M_I1
    for i in range (j-2):
        I_j = sparse.kron(I_j , M_I2)
    for i in range(n-k):
        I_nk = sparse.kron(I_nk,M_I2)
    if j<k:
        mcgate = sparse.kron(I_j, sparse.kron(make_mcgate_1n(k-j+1, P0, P1, gate),I_nk))
    else:
        mcgate = sparse.kron(I_j, sparse.kron(make_mcgate_n1(j-k+1, P0, P1, gate),I_nk))
    return mcgate;

def from_Q_to_C(gate):
    ReGate = np.real(gate)
    ImGate = np.imag(gate)
    M_base_re = np.eye(4)
    M_base_img =np.array([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]])
    M_gate = sparse.kron(M_base_re, ReGate) + sparse.kron(M_base_img, ImGate)
    return sparse.csr_matrix(M_gate)

def make_swapmcgate_1n(n,P0,P1):
    NOT = np.array([[0, 1],
                    [1, 0]])
    NOT1n = make_mcgate_1n(n,P0,P1,NOT)
    NOTn1 = make_mcgate_n1(n,P0,P1,NOT)
    SWAP1n = NOT1n.dot(NOTn1).dot(NOT1n)
    return SWAP1n

def make_swapmcgate_n1(n,P0,P1):
    NOT = np.array([[0, 1],
                    [1, 0]])
    NOT1n = make_mcgate_1n(n,P0,P1,NOT)
    NOTn1 = make_mcgate_n1(n,P0,P1,NOT)
    SWAPn1 = NOTn1.dot(NOT1n).dot(NOTn1)
    return SWAPn1

def make_swapmcgate_jk(n,j,k,P0,P1):
    NOT = np.array([[0, 1],
                    [1, 0]])
    NOTjk = make_mcgate_jk(n,j,k,P0,P1,NOT)
    NOTkj = make_mcgate_jk(n,k,j,P0,P1,NOT)
    SWAPjk = NOTjk.dot(NOTkj).dot(NOTjk)
    return SWAPjk

def make_MP(phi):
    phase = np.exp(1j * phi)
    
    # 4x4 CPHASE gate matrix
    CPHASE = np.array([[1,0],
                      [0,phase]])
    
    return CPHASE

#def algo_make_gates(n):
    
#Example:

CP2 =  make_MP(np.pi/2)

# =============================================================================
# MCSWAP23 = make_swapmcgate_jk(4,2,3, P0, P1) 
# make_spars(MCSWAP23)
# MCSWAP32 = make_swapmcgate_jk(4,3,2, P0, P1) 
# make_spars(MCSWAP32)
# MCSWAP14 = make_swapmcgate_1n(4, P0, P1) 
# make_spars(MCSWAP23)
# MCSWAP41 = make_swapmcgate_n1(4, P0, P1) 
# make_spars(MCSWAP23)
# =============================================================================

# =============================================================================
# MCNOT41 = make_mcgate_n1(4,P0,P1,NOT)
# make_spars(MCNOT41)
# MCNOT14 = make_mcgate_1n(4,P0,P1,NOT)
# make_spars(MCNOT14)
# MCNOT32 = make_mcgate_jk(4,3,2,P0,P1,NOT)
# make_spars(MCNOT32)
# MCNOT23 = make_mcgate_jk(4,2,3,P0,P1,NOT)
# make_spars(MCNOT23)
# =============================================================================

MCP_21 = make_mcpgate_jk(4, 1, 2, P0, P1,CP2)
MCP_41 = make_mcgate_1n(4, P0, P1,make_MP(np.pi/8))
MCP_31 = make_mcpgate_jk(4, 1, 3, P0, P1, make_MP(np.pi/4))
MCP_23 = make_mcpgate_jk(4, 2, 3, P0, P1,CP2)
MCP_24 = make_mcpgate_jk(4, 2, 4, P0, P1,make_MP(np.pi/4))
MCP_13 = make_mcpgate_jk(4, 1, 3, P0, P1,make_MP(np.pi/4))


