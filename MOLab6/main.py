import numpy as np

def dummy_variable(self, params, function):
    A_extended = np.hstack((self, np.eye(self.shape[0])))
    c_extended = np.concatenate((function, np.zeros(self.shape[0])))
    A_extended = np.vstack((A_extended, c_extended))
    b_extended = np.append(params, 0)
    A_extended = np.column_stack((A_extended, b_extended))
    return A_extended


def norm_c(matrix):
    max_element_row_index = np.argmax(matrix[-1, :-1])
    max_element_column = np.max(matrix[:, max_element_row_index])
    max_element_column_index = np.argmax(matrix[:, max_element_row_index])

    matrix[max_element_column_index, :] /= max_element_column
    for i in range(matrix.shape[0]):
        if i != max_element_column_index:
            ratio = matrix[i, max_element_row_index] / matrix[max_element_column_index, max_element_row_index]
            matrix[i, :] -= ratio * matrix[max_element_column_index, :]

    return matrix


def norm_b(matrix):
    max_element_column_index = np.argmin(matrix[:-1, -1])
    max_element_row_index = np.argmin(matrix[max_element_column_index, :-1])
    max_element_row = matrix[max_element_column_index, max_element_row_index]

    matrix[max_element_column_index, :] /= max_element_row
    for i in range(matrix.shape[0]):
        if i != max_element_column_index:
            ratio = matrix[i, max_element_row_index] / matrix[max_element_column_index, max_element_row_index]
            matrix[i, :] -= ratio * matrix[max_element_column_index, :]

    return matrix


def simplex_max(matrix):
    i = 1
    while np.any(matrix[-1, :-1] > 0):
        norm_c(matrix)

        print(f"\nШаг №{i}. Нормированная матрица:")
        print_matrix(matrix)
        i += 1

    basic_variables = []
    for col in range(matrix.shape[1] - 1):
        if abs(np.all(matrix[:, col] == 0)) or abs(np.count_nonzero(matrix[:, col])) != 1:
            basic_variables.append(0)
        else:
            row_index = np.argmax(np.abs(matrix[:, col]))
            basic_variables.append(matrix[row_index, -1])

    optimal_value = matrix[-1, -1]
    return basic_variables, optimal_value


def simplex_min(matrix):
    i = 1
    while np.any(matrix[:-1, -1] < 0):
        norm_b(matrix)

        print(f"\nШаг №{i}. Нормированная матрица:")
        print_matrix(matrix)
        i += 1

    basic_variables = []
    for col in range(matrix.shape[1] - 1):
        if abs(np.all(matrix[:, col] == 0)) or abs(np.count_nonzero(matrix[:, col])) != 1:
            basic_variables.append(0)
        else:
            row_index = np.argmax(np.abs(matrix[:, col]))
            basic_variables.append(matrix[row_index, -1])

    optimal_value = matrix[-1, -1]
    return basic_variables, optimal_value


def print_matrix(matrix):
    width = max(len(f"{value:.2f}") for row in matrix for value in row)

    for row in matrix:
        for value in row:
            print(f"{value:>{width}.2f}", end="\t")
        print()


A = np.array([
    [7, 9, 15, 5],
    [15, 8, 6, 4],
    [12, 0, 11, 7],
    [7, 11, 10, 12],
    [12, 2, 0, 13]
])
print(A)
c = np.array([1, 1, 1, 1])
b = np.array([1, 1, 1, 1, 1])
print("Симплекс-таблица для игрока A:")
print_matrix(dummy_variable(-1 * np.transpose(A), -1 * c, -1 * b))

result_variables_A, result_value_A = simplex_min(dummy_variable(-1 * np.transpose(A), -1 * c, -1 * b))

print("\nОптимальное значение переменных:")
print(f"u1 = {round(result_variables_A[0], 4)} \nu2 = {round(result_variables_A[1], 4)} \nu3 = {round(result_variables_A[2], 4)} \nu4 = {round(result_variables_A[3], 4)} \nu5 = {round(result_variables_A[4], 4)}")

print("W =", round(result_value_A, 4))


print("\nСимплекс-таблица для игрока В:")
print_matrix(dummy_variable(A, b, c))

result_variables_B, result_value_B = simplex_max(dummy_variable(A, b, c))

print("\nОптимальное значение переменных:")
print(f"v1 = {round(result_variables_B[0], 4)} \nv2 = {round(result_variables_B[1], 4)} \nv3 = {round(result_variables_B[2], 4)} \nv4 = {round(result_variables_B[3], 4)}")

print("Z =", round(-result_value_B, 4))


g = 1 / result_value_A
print("-----------------------\ng = ", round(g, 4))
print(f"x1 = {round(g * result_variables_A[0], 4)} \nx2 = {round(g * result_variables_A[1], 4)} \nx3 = {round(g * result_variables_A[2], 4)} \nx4 = {round(g * result_variables_A[3], 4)} \nx5 = {round(g * result_variables_A[4], 4)}")
print(f"Оптимальная смешанная стратегия игрока А - ({round(g * result_variables_A[0], 4)}, {round(g * result_variables_A[1], 4)}, {round(g * result_variables_A[2], 4)}, {round(g * result_variables_A[3], 4)}, {round(g * result_variables_A[4], 4)})")

h = -1 / result_value_B
print("-----------------------\nh = ", round(h, 4))
print(f"y1 = {round(h * result_variables_B[0], 4)} \ny2 = {round(h * result_variables_B[1], 4)} \ny3 = {round(h * result_variables_B[2], 4)} \ny4 = {round(h * result_variables_B[3], 4)}")
print(f"Оптимальная смешанная стратегия игрока В - ({round(h * result_variables_B[0], 4)}, {round(h * result_variables_B[1], 4)}, {round(h * result_variables_B[2], 4)}, {round(h * result_variables_B[3], 4)})")

print("\nЦена игры будет равна:\n1/W = 1/Z =", round(g, 4))

