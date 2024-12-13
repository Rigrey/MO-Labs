import branches
import bruteforce
def main():
    c = [7, 7, 6]
    A = [[2, 1, 1],
         [1, 2, 0],
         [0, 0.5, 4]]
    b = [8, 2, 6]
    f = 0
    minimize = False

    print("Best answer: ", branches.method_execute(c, A, b, f, minimize))
    print("Brute-force: ", bruteforce.method_execute(c, A, b, f, minimize))

if __name__ == '__main__':
    main()