import numpy as np 

def ctrb_mat(A, B):
    """ Computes the controllability matrix 
        C = [B AB A^2B ... A^(n-1)B] 
    """
    C=[B]
    n = A.shape[0]
    
    for _ in range(1, n):
        C.append(A.dot(C[-1]))
        
    return np.array(C)   
    
def ctrb(A, B):
    """ Determines if (A,B) is controllable via rank condition """
    n = A.shape[0]
    C = ctrb_mat(A, B)
    r = np.linalg.matrix_rank(C)
    return r==n
        
if __name__ =="__main__":

    A = np.array([[0,1,1],[-1,-1,0],[0,0,0]])
    B = [0,0,1]     # nxm
    
    C = ctrb_mat(A, B)   
    print(C.shape)   # = nxnm
    print(ctrb(A, B))