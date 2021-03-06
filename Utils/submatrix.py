import numpy as np

def submatrices(V, rows, cols=None):
    """ Performs submatrix operation on each matrix in the list V """
    return np.array([submatrix(M, rows, cols) for M in V])

def submatrix(M, rows, cols=None):
    """ From a matrix M returns the elements M[rows,cols] """
    if cols is None:
        cols = rows
    M = np.asarray(M)
    S = M[rows, :][:, cols]

    return S


def test_submatrix():

    matrix = np.random.random((5, 5))
    print(matrix)
    sub = submatrix(matrix, [0, 2], [0, 2])
    print(sub)


if __name__ == "__main__":
    test_submatrix()
