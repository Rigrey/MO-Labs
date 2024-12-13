columns = []
rows = []
old_columns = []

def init_headers(c, A, b):
    num_columns = len(c)
    num_rows = len(b)
    global columns
    global rows
    global old_columns
    columns = ['b'] + [f'x{i + 1}' for i in range(num_columns)]
    old_columns = columns.copy()
    rows = []
    for i in range(num_rows):
        temp_row = [f'x{i + num_columns + 1}']
        rows.append(temp_row)
    rows.append('F')

def validate_simplex_input(c, A, b):
    len_A = len(A[0])
    return (all(len(row) == len_A for row in A) and
            len(c) == len_A and
            len(b) == len(A))

def check_solution_existence(c, A, b):
    if all(x == 0 for x in c):
        return False
    return all(b[row] >= 0 or min(A[row]) < 0 for row in range(len(b)))

def construct_simplex_table(c, A, b, f):
    table = []
    for i in range(len(A)):
        table.append([b[i]] + A[i])
    table.append([f] + c)
    return table

def format_header_value(value):
    return " ".join(str(v) for v in value) if isinstance(value, list) else str(value)

def display_simplex_table(simplex_table):
    print()
    formatted_columns = [format_header_value(col) for col in columns]
    formatted_rows = [format_header_value(row) for row in rows]
    max_width = max(len(str(float(j))) for row in simplex_table for j in row if isinstance(j, (int, float))) + 2
    full_headers = [""] + formatted_columns  # Добавляем 'b' перед заголовками столбцов
    print(" | ".join(f"{header:>{max_width}}" for header in full_headers))
    print("-" * (max_width * len(full_headers) + 3 * (len(full_headers) - 1)))
    for i, row in enumerate(simplex_table):
        row_values = [formatted_rows[i]] + list(row)  # Добавляем заголовок для строки 'b'
        print(" | ".join(
            f"{float(j):>{max_width}.2f}" if isinstance(j, (int, float)) else str(j).ljust(max_width) for j in
            row_values))

def locate_resolving_element(c, A, b):
    if not check_solution_existence(c, A, b):
        return ["not_er"]
    for row in range(len(b)):
        if b[row] < 0:
            for col in range(len(A[0])):
                if A[row][col] < 0:
                    try:
                        return calculate_min_ratio(A, b, col)
                    except:
                        return ["inf_er"]
    if max(c) < 0:
        return ["not_er"]
    c_max_index = c.index(max(c))
    return calculate_min_ratio(A, b, c_max_index)

def calculate_min_ratio(A, b, ratio_col):
    min_ratio = float("inf")
    min_ratio_row = -1
    for row in range(len(A)):
        if A[row][ratio_col] == 0:
            continue
        ratio = b[row] / A[row][ratio_col]
        if 0 < ratio < min_ratio:
            min_ratio = ratio
            min_ratio_row = row
    if min_ratio_row == -1:
        raise ValueError("No valid resolving element found.")
    return [A[min_ratio_row][ratio_col], min_ratio_row, ratio_col]

def perform_simplex_iteration(c, A, b, f, res_el):
    global columns, rows
    new_resolving_element = 1 / res_el[0]
    new_b = [0] * len(b)
    new_A = [[0] * len(A[0]) for _ in A]
    for i in range(len(A)):
        new_A[i][res_el[2]] = (
            new_resolving_element if i == res_el[1]
            else A[i][res_el[2]] / res_el[0] * -1
        )

    new_c = [0] * len(c)
    new_c[res_el[2]] = c[res_el[2]] / res_el[0] * -1
    for i in range(len(A[0])):
        if i != res_el[2]:
            new_A[res_el[1]][i] = A[res_el[1]][i] / res_el[0]

    new_b[res_el[1]] = b[res_el[1]] / res_el[0]
    for i in range(len(c)):
        if i == res_el[2]:
            continue
        new_c[i] = c[i] - (A[res_el[1]][i] * c[res_el[2]]) / (res_el[0])

    for i in range(len(b)):
        if i == res_el[1]:
            continue
        new_b[i] = b[i] - ((A[i][res_el[2]] * b[res_el[1]]) / res_el[0])

    for i in range(len(A)):
        for j in range(len(A[0])):
            if (i == res_el[1]) or (j == res_el[2]):
                continue
            new_A[i][j] = A[i][j] - ((A[i][res_el[2]] * A[res_el[1]][j]) / res_el[0])

    new_f = f - ((c[res_el[2]] * b[res_el[1]]) / res_el[0])
    temp_row = rows[res_el[1]]
    rows[res_el[1]] = columns[1+res_el[2]]
    columns[1+res_el[2]] = temp_row
    return new_c, new_A, new_b, new_f

def execute_simplex(c, A, b, f, minimize):
    if validate_simplex_input(c, A, b):
        print("Input Validation: Passed")
        init_headers(c, A, b)
        if minimize:
            c = [-x for x in c]
        while (max(c) > 0) or (min(b) < 0):
            simplex_table = construct_simplex_table(c, A, b, f)
            display_simplex_table(simplex_table)
            resolving_element = locate_resolving_element(c, A, b)
            if resolving_element == ["not_er"]:
                print("No feasible solution exists.")
                return [float("inf")]
            if resolving_element == ["inf_er"]:
                print("Infinite solutions detected.")
                return [float("inf")]
            print("Resolving element located:", resolving_element)
            c, A, b, f = perform_simplex_iteration(c, A, b, f, resolving_element)

        print("\nOptimal Solution Found:")
        simplex_table = construct_simplex_table(c, A, b, f)
        display_simplex_table(simplex_table)
    else:
        print("Input Validation: Failed")
        return [float("inf")]

    if minimize:
        print("The function is minimized.")
        return [f] + create_answer_variables(b, rows, old_columns)
    print("The function is maximized.")
    return [f * -1] + create_answer_variables(b, rows, old_columns)

def create_answer_variables(b, var_row, old_var_col):
    answer_variables = [0 for _ in range(len(old_var_col) - 1)]
    for i in range(len(var_row)):
        if var_row[i] in old_var_col:
            answer_variables[old_var_col.index(var_row[i]) - 1] = round(b[i], 2)  # Сохранение значения
    return answer_variables