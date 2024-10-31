from simp_solv import execute_simplex
from dual_solv import convert_to_dual_problem

def main():
    c = [7, 7, 6]
    A = [[2, 1, 1],
         [1, 2, 0],
         [0, 0.5, 4]]
    b = [8, 2, 6]
    f = 0
    minimize = False
    dual_c, dual_A, dual_b, dual_min = convert_to_dual_problem(c, A, b, minimize)
    print("Dual Answer:", execute_simplex(dual_c, dual_A, dual_b, f, dual_min))


if __name__ == '__main__':
    main()