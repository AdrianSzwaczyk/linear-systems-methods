import math
import matplotlib.pyplot as plt
from matrix_math import *
from methods import *

def plot_convergence(residuals_j, residuals_gs, filename = 'plot.png'):
    iterations_j = list(range(1, len(residuals_j) + 1))
    iterations_gs = list(range(1, len(residuals_gs) + 1))
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations_j, residuals_j, marker='o', linestyle='-', label='Jacobi Method')
    plt.semilogy(iterations_gs, residuals_gs, marker='o', linestyle='-', label='Gauss-Seidel Method')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence of Iterative Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filename}.png')
    plt.show()

# Task A - Create the system of equations
def create_system_of_equations(N, a1, a2, a3, f):
    b = [math.sin(n * (f + 1)) for n in range(1, N + 1)]
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i][j] = a1
            elif abs(i - j) == 1:
                A[i][j] = a2
            elif abs(i - j) == 2:
                A[i][j] = a3
    return A, b

# Task B - Jacobi and Gauss-Seidel methods
def task_b(A, b):
    time_j, x_j, residuals_j = jacobi_method(A, b)
    print("Jacobi Method Time:", time_j)
    time_gs, x_gs, residuals_gs = gauss_seidel_method(A, b)
    print("Gauss-Seidel Method Time:", time_gs)
    plot_convergence(residuals_j, residuals_gs, 'task_b')

# Task C - Check convergence of iterative methods
def task_c():
    A, b = create_system_of_equations(933, 3, -1, -1, 3)

    time_j, x_j, residuals_j = jacobi_method(A, b)
    print("Jacobi Method Time:", time_j)
    time_gs, x_gs, residuals_gs = gauss_seidel_method(A, b)
    print("Gauss-Seidel Method Time:", time_gs)
    plot_convergence(residuals_j, residuals_gs, 'task_c')

# Task D - LU decomposition
def task_d(A, b):
    A, b = create_system_of_equations(933, 3, -1, -1, 3)
    _, LU = lu_decomposition(A)
    n = len(b)
    
    y = [0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= LU[i][j] * y[j]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= LU[i][j] * x[j]
        x[i] /= LU[i][i]
    
    r = [0] * n
    for i in range(n):
        r[i] = b[i] - sum(A[i][j] * x[j] for j in range(n))

    residual_norm = sum(val**2 for val in r)**0.5
    print(residual_norm)
    return x, residual_norm

# Task E - Time complexity analysis
def task_e(A, b):
    N_values = [100, 500, 1000, 2000, 3000]
    times_j = []
    times_gs = []
    times_lu = []
    for N in N_values:
        A, b = create_system_of_equations(N, 5 + 2, -1, -1, 3)
        time_j, _, _ = jacobi_method(A, b)
        times_j.append(time_j)
        
        A, b = create_system_of_equations(N, 5 + 2, -1, -1, 3)
        time_gs, _, _ = gauss_seidel_method(A, b)
        times_gs.append(time_gs)
        
        A, b = create_system_of_equations(N, 5 + 2, -1, -1, 3)
        time_lu, _ = lu_decomposition(A)
        times_lu.append(time_lu)
        
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, times_j, marker='o', linestyle='-', label='Jacobi Method')
    plt.plot(N_values, times_gs, marker='o', linestyle='-', label='Gauss-Seidel Method')
    plt.plot(N_values, times_lu, marker='o', linestyle='-', label='LU Decomposition')
    plt.xlabel('Number of Unknowns (N)')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('task_e.png')
    plt.show()
    return

# TASK A
A, b = create_system_of_equations(933, 5 + 2, -1, -1, 3)

# TASK B
task_b(A, b)

# TASK C
task_c()

# TASK D
task_d(A, b)

# TASK E
task_e(A, b)