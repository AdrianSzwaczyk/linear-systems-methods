def vector_multiply(matrix, vector):
    result = []
    for i in range(len(matrix)):
        row_sum = 0
        for j in range(len(vector)):
            row_sum += matrix[i][j] * vector[j]
        result.append(row_sum)
    return result