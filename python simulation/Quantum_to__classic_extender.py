import numpy as np



#Quantum matrix of CP
def QCP(n,phi,q1,q2):
    l=2**n
    CP = np.diag(np.ones(l))

    #For each row check if both relevant bits values are 1 and
    #change the value in the matrix to the phase change
    for i in range(l):
        b=f'{i:0{n}b}'
        if b[q1]==1 and b[q2]==1:
            CP[i,i]=np.exp(1j*phi)

    return CP

#Classic matrix of CP
def CCP(n,phi,q1,q2):
    l=2**n
    M=QCP(n,phi,q1,q2)
    RM = np.real(M)
    IM = np.imag(M)

    zeroM = np.zeros([l, l])
    C = np.block([[RM, zeroM, zeroM, IM],
                [zeroM, RM, IM, zeroM],
                [IM, zeroM, RM, zeroM],
                [zeroM, IM, zeroM, RM]])
    return C

#Quantum matrix of SWAP
def QSWAP(n,q1,q2):
    if q1>q2:
        q1,q2=q2,q1

    l=2**n
    M=np.zeros([l,l])
    # For each row value swap the relevant bits values  and
    # change the value in the corresponding position to 1
    b=0
    for i in range(l):
        b=f'{i:0{n}b}'
        rb=b[:q1]+b[q2]+b[q1+1:q2]+b[q1]+b[q2+1:]
        ri=int(rb, 2)
        #print(b,rb,ri)
        M[ri,i]=1
    return M

#Classic matrix of SWAP
def CSWAP(n,q1,q2):
    M=QSWAP(n,q1,q2)
    C=np.kron(np.eye(4),M)
    return C

#Quantum matrix of H
def QH(n,ql):
    l=2**n
    #Calculate the tensor product of all qubits (H for qubits
    #that H is applied to, and I to those that it isn't)
    H=1/np.sqrt(2)*np.array([[1,1],
                             [1,-1]])
    I=np.eye(2)
    if 0 in ql:
        M=H
    else:
        M=I

    for i in range(1,n):
        if i in ql:
            M=np.kron(M,H)
        else:
            M=np.kron(M,I)
    return M

#Classic matrix of H
def CH(n,ql):
    M=QH(n,ql)
    C = np.kron(np.eye(4),M)
    return C


def QCNOT(n,q1,q2):
    l = 2 ** n
    CN = np.diag(np.ones(l))
    for i in range(l):
        b=f'{i:0{n}b}'
        if b[q1]=='1':
            CN[i,i]=0
            temp=1-int(b[q2])
            b=b[:q2]+str(temp)+b[q2+1:]
            row=int(b,2)
            CN[row,i]=1

    return CN

#Classic matrix of CNOT
def CCNOT(n,q1,q2):
    M=QCNOT(n,q1,q1)
    C = np.kron(np.eye(4), M)
    return C

def Q2C(M):
    l=M.shape[0]
    RM = np.real(M)
    IM = np.imag(M)

    zeroM = np.zeros([l, l])
    C = np.block([[RM, zeroM, zeroM, IM],
                  [zeroM, RM, IM, zeroM],
                  [IM, zeroM, RM, zeroM],
                  [zeroM, IM, zeroM, RM]])
    return C

if __name__=='__main__':
    print(QCP(np.pi/2,2, 0 ,0))
    print(QSWAP(3,0,1))
    print(QH(2,[0,1]))
    print(CH(1,[0]))
    print(QCNOT(3,0,1))
    print(Q2C(QH(1,[0])))
    None