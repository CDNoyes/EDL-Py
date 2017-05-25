import numpy as np

def submatrix(M,rows,cols):
    """ From a matrix M returns the elements M[rows,cols] """
    M = np.asarray(M)
    S = M[rows,:][:,cols]

    return S

def test_submatrix():
    
    matrix = np.random.random((5,5))
    print matrix
    sub = submatrix(matrix,[0,2],[0,2])
    print sub 
    
    
if __name__ == "__main__":
    test_submatrix() 