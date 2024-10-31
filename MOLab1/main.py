from simp_solv import execute_simplex

def main():
    minimize = False
    c = [7, 7, 6]
    A = [[2, 1, 1],
         [1, 2, 0],
         [0, 0.5, 4]]
    b = [8, 2, 6]
    f = 0
    print("Answer:", execute_simplex(c, A, b, f, minimize))


if __name__ == "__main__":
    main()