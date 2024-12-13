from simp_solv import execute_simplex
from math import floor


def method_execute(c, A, b, f, minimize, best_solution=None):
    stack = [(c, A, b, f, minimize, best_solution)]

    while stack:
        current_c, current_A, current_b, current_f, current_minimize, current_best_solution = stack.pop()
        answer_simplexsus = execute_simplex(current_c, current_A, current_b, current_f, current_minimize)
        if answer_simplexsus[0] == float("inf"):
            print("Ветвь не имеет решения")
            continue
        answer_variables = answer_simplexsus[1::]
        current_best_solution = check_integer_solution(answer_simplexsus, answer_variables, current_best_solution)
        if current_best_solution is not None and (best_solution is None or current_best_solution[0] > best_solution[0]):
            best_solution = current_best_solution
        i = 0
        found = False
        while i < len(answer_variables) and not found:
            if not is_integer(answer_variables[i]):
                branching_variable = floor(answer_variables[i])
                print(
                    f"Adding a new condition for x{i + 1} = {answer_variables[i]}:",
                    f"[x{i + 1} <= {branching_variable}; x{i + 1} >= {branching_variable + 1}]",
                )

                # Создаем новые ограничения
                new_A_left = current_A + [[0 if j != i else 1 for j in range(len(c))]]
                new_b_left = current_b + [branching_variable]

                # Добавляем новое ограничение для x_i <= branching_variable
                stack.append((current_c, new_A_left, new_b_left, current_f, current_minimize, best_solution))

                # Добавляем ограничение для x_i >= branching_variable + 1
                new_A_right = current_A + [[0 if j != i else -1 for j in range(len(c))]]
                new_b_right = current_b + [(branching_variable + 1) * -1]

                stack.append((current_c, new_A_right, new_b_right, current_f, current_minimize, best_solution))

                found = True

            i += 1

    return best_solution


def check_integer_solution(answer_simplexsus, answer_variables, best_solution):
    if all(is_integer(var) for var in answer_variables):
        if best_solution is None or answer_simplexsus[0] < best_solution[0]:
            best_solution = answer_simplexsus
            print("New best solution found:", best_solution, "\n")
        else:
            print("Current solution is integer but not better:", answer_simplexsus)

    return best_solution


def is_integer(value):
    return floor(value) == value
