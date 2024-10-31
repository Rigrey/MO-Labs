def convert_to_dual_problem(c, A, b, minimize):
    dual_c = b
    dual_b = [-x for x in c]
    dual_A = [[-A[row][col] for row in range(len(A))] for col in range(len(A[0]))]
    return dual_c, dual_A, dual_b, not minimize