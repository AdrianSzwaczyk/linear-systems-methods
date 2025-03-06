import time
from matrix_math import * 

def jacobi_method(A, b, max_iterations=1000, tolerance=1e-9, max_residual_norm=1e9):
    start_time = time.time()
    n = len(A)
    x = [0] * n
    x_new = [0] * n
    residuals = []
    for _ in range(max_iterations):
        for i in range(n):
            sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_ax) / A[i][i]
        residual_norm = sum((A_row_x - b_row)**2 for A_row_x, b_row in zip(vector_multiply(A, x_new), b))**0.5
        residuals.append(residual_norm)
        if residual_norm < tolerance:
            return time.time() - start_time, x_new, residuals
        elif residual_norm > max_residual_norm:
            return time.time() - start_time, x_new, residuals
        x = x_new[:]
    return time.time() - start_time, x_new, residuals

def gauss_seidel_method(A, b, max_iterations=1000, tolerance=1e-9, max_residual_norm=1e9):
    start_time = time.time()
    n = len(A)
    x = [0] * n
    residuals = []
    for _ in range(max_iterations):
        for i in range(n):
            sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_ax) / A[i][i]
        residual_norm = sum((A_row_x - b_row)**2 for A_row_x, b_row in zip(vector_multiply(A, x), b))**0.5
        residuals.append(residual_norm)
        if residual_norm < tolerance:
            return time.time() - start_time, x, residuals
        elif residual_norm > max_residual_norm:
            return time.time() - start_time, x, residuals
    return time.time() - start_time, x, residuals


def lu_decomposition(A):
    start_time = time.time()
    n = len(A)
    LU = [row[:] for row in A]
    
    for k in range(n-1):
        for i in range(k+1, n):
            LU[i][k] /= LU[k][k]
            for j in range(k+1, n):
                LU[i][j] -= LU[i][k] * LU[k][j]
    
    return time.time() - start_time, LU