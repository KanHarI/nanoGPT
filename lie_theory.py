import torch


def skew_symmetric_matrix(n, algebra_vector):
    matrix = torch.zeros(n, n)

    # Get the indices of the lower triangular part
    row_indices, col_indices = torch.tril_indices(n, n, -1)

    # Assign values to the lower triangular part
    matrix.index_put_((row_indices, col_indices), -algebra_vector)

    # Assign values to the upper triangular part
    matrix.index_put_((col_indices, row_indices), algebra_vector)

    return matrix


def so_to_SO(n, algebra_vector):
    # Convert algebra_vector to a PyTorch tensor
    algebra_vector = torch.tensor(algebra_vector, dtype=torch.float32)

    # Create skew-symmetric matrix
    matrix_1 = skew_symmetric_matrix(n, algebra_vector)

    # Compute the matrix exponential of matrix_1
    print(matrix_1)
    matrix_2 = torch.matrix_exp(matrix_1)
    print(matrix_2)

    # Serialize the result into a vector
    result = matrix_2.view(-1).tolist()

    return result
