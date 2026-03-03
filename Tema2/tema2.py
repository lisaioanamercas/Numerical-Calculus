"""
Tema 2 - Calcul Numeric
Descompunere LDLT (Cholesky) pentru sisteme liniare cu matrice simetrica pozitiv definita.

Cerinta:
- Calcul descompunere LU folosind biblioteca (numpy)
- Calcul descompunere LDLT (Cholesky) manual
- Calcul determinant folosind descompunerea LDLT
- Rezolvarea sistemului Ax = b prin substitutie directa/inversa
- Verificare solutie prin norme euclidiene
- Restrictie: o singura matrice A, D memorat ca vector
"""

import numpy as np
import scipy.linalg as la
import math


def generate_spd_matrix(n: int) -> np.ndarray:
    """
    Genereaza o matrice simetrica pozitiv definita de dimensiune n x n.
    Se genereaza mai intai o matrice B aleatoare, apoi A = B @ B^T.
    """
    B = np.random.rand(n, n)
    A = B @ B.T
    # Adaugam n*I pentru a fi siguri ca A este strict pozitiv definita
    A += n * np.eye(n)
    return A


def cholesky_ldlt(A: np.ndarray, eps: float):
    """
    Calculeaza descompunerea LDLT a matricei A simetrice pozitiv definite.

    Restrictie: modifica matricea A in-place:
      - Partea strict inferior triunghiulara a lui A va contine elementele L (sub diagonala)
      - Partea superior triunghiulara ramane neschimbata (elementele initiale ale lui A)
      - Diagonala lui D este returnata separat ca vector d

    Returneaza:
      - A modificata (in-place)
      - d: vectorul diagonalei lui D
    """
    n = A.shape[0]
    d = np.zeros(n)

    for p in range(n):
        # Pasul p: calculeaza d[p] si coloana p a lui L (elementele sub diagonala)

        # Calcul d[p] = a[p,p] - sum(d[k] * l[p,k]^2, k=0..p-1)
        # l[p,k] pentru k < p se afla deja in A[p, k] (zona strict inf triunghiulara)
        dp = A[p, p]
        for k in range(p):
            dp -= d[k] * A[p, k] ** 2  # A[p,k] = l[p,k]

        if math.fabs(dp) <= eps:
            raise ValueError(f"Elementul diagonal d[{p}] = {dp} este prea mic (< eps). "
                             "Matricea nu este pozitiv definita sau algoritmul a esuat.")
        d[p] = dp

        # Calcul l[i,p] pentru i = p+1, ..., n-1
        # l[i,p] = (a[i,p] - sum(d[k]*l[i,k]*l[p,k], k=0..p-1)) / d[p]
        for i in range(p + 1, n):
            s = A[i, p]  # a[i,p] original (din zona superior triunghiulara a lui A: A[p,i])
            # Folosim simetria: elementul original a[i,p] = a[p,i] se afla in A[p, i]
            # (zona superior triunghiulara nu a fost modificata)
            s = A[p, i]  # a_init[i,p] = a_init[p,i] prin simetrie
            for k in range(p):
                s -= d[k] * A[i, k] * A[p, k]  # A[i,k]=l[i,k], A[p,k]=l[p,k]
            A[i, p] = s / d[p]  # Scriem l[i,p] in zona strict inf triunghiulara

    return A, d


def forward_substitution_unit_diag(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Rezolva sistemul Lz = b unde L este inferior triunghiulara cu 1 pe diagonala.
    Elementele lui L (sub diagonala) se afla in zona strict inf triunghiulara a lui A.
    Formula: z[i] = b[i] - sum(l[i,j]*z[j], j=0..i-1)
    """
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= A[i, j] * z[j]  # A[i,j] = l[i,j] pentru j < i
        z[i] = s  # l[i,i] = 1, nu impartim
    return z


def diagonal_solve(d: np.ndarray, z: np.ndarray, eps: float) -> np.ndarray:
    """
    Rezolva sistemul Dy = z unde D = diag(d).
    Formula: y[i] = z[i] / d[i]
    """
    n = len(z)
    y = np.zeros(n)
    for i in range(n):
        if math.fabs(d[i]) <= eps:
            raise ValueError(f"d[{i}] = {d[i]} prea mic, nu se poate imparti.")
        y[i] = z[i] / d[i]
    return y


def backward_substitution_unit_diag(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Rezolva sistemul L^T x = y unde L^T este superior triunghiulara cu 1 pe diagonala.
    Elementele L^T[i,j] = L[j,i] = A[j,i] pentru j > i (din zona strict inf triunghiulara a lui A).
    Formula: x[i] = y[i] - sum(l[j,i]*x[j], j=i+1..n-1)
           = y[i] - sum(A[j,i]*x[j], j=i+1..n-1)
    """
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= A[j, i] * x[j]  # L^T[i,j] = L[j,i] = A[j,i]
        x[i] = s  # l[i,i] = 1
    return x


def compute_Ainit_times_x(A_modified: np.ndarray, d: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculeaza produsul Ainit @ x fara a folosi procedura de inmultire matrice-vector din biblioteca.
    Matricea A a fost modificata de algoritmul Cholesky:
      - Zona superior triunghiulara (inclusiv diagonala) contine elementele originale ale lui A
      - Zona strict inferior triunghiulara contine elementele lui L

    Folosim simetria: a_init[i,j] = A_modified[i,j] pentru j >= i (superior triunghiulara)
                      a_init[i,j] = a_init[j,i] = A_modified[j,i]  -- NU, zona asta e L!

    Corect: elementele originale a_init[i,j] cu j < i (sub diagonala) se gasesc la A_modified[j, i]
    (prin simetrie a_init[i,j] = a_init[j,i], iar a_init[j,i] cu j < i este in zona superior
    triunghiulara, deci la A_modified[j, i]).
    """
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j >= i:
                # elementul original a[i,j] se afla in zona superior triunghiulara
                a_ij = A_modified[i, j]
            else:
                # a[i,j] = a[j,i] (simetrie), iar a[j,i] cu j < i, i > j => j < i
                # deci a[j,i] este in zona superior triunghiulara la pozitia A_modified[j, i]
                a_ij = A_modified[j, i]
            s += a_ij * x[j]
        y[i] = s
    return y


def compute_determinant(d: np.ndarray) -> float:
    """
    Calculeaza determinantul matricei A folosind descompunerea LDLT.
    det(A) = det(L) * det(D) * det(L^T) = 1 * prod(d) * 1 = prod(d)
    (det(L) = det(L^T) = 1 deoarece sunt matrice triunghiulare cu 1 pe diagonala)
    """
    det = 1.0
    for di in d:
        det *= di
    return det


def solve_with_cholesky(A_orig: np.ndarray, b: np.ndarray, eps: float):
    """
    Rezolva sistemul Ax = b folosind descompunerea LDLT (Cholesky).
    Returneaza solutia xChol, determinantul si A modificata (cu L stocat in zona inf).
    """
    n = A_orig.shape[0]
    # Copiem A pentru a lucra in-place (o singura matrice!)
    A = A_orig.copy()

    # 1. Calcul descompunere LDLT
    A, d = cholesky_ldlt(A, eps)

    # 2. Calcul determinant
    det_A = compute_determinant(d)

    # 3. Rezolvare Lz = b (substitutie directa, L cu 1 pe diagonala)
    z = forward_substitution_unit_diag(A, b)

    # 4. Rezolvare Dy = z
    y = diagonal_solve(d, z, eps)

    # 5. Rezolvare L^T x = y (substitutie inversa, L^T cu 1 pe diagonala)
    x_chol = backward_substitution_unit_diag(A, y)

    return x_chol, det_A, A, d


def main():
    print("=" * 60)
    print("TEMA 2 - Descompunere LDLT (Cholesky)")
    print("=" * 60)

    # Citire parametri
    try:
        n = int(input("\nIntroduceti dimensiunea sistemului n: "))
        t = int(input("Introduceti precizia t (eps = 10^(-t), ex: t=8): "))
    except ValueError:
        print("Input invalid. Se folosesc valori implicite: n=5, t=8.")
        n, t = 5, 8

    eps = 10 ** (-t)
    print(f"\nn = {n}, eps = {eps:.2e}")

    # Generare matrice A simetrica pozitiv definita si vector b
    np.random.seed(42)
    A_init = generate_spd_matrix(n)
    b = np.random.rand(n)

    print("\n--- Matricea A (initiala) ---")
    print(np.array2string(A_init, precision=4, suppress_small=True))
    print("\n--- Vectorul b ---")
    print(np.array2string(b, precision=6))

    # -------------------------------------------------------
    # 1. Descompunere LU si solutie prin biblioteca (numpy/scipy)
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. DESCOMPUNERE LU (biblioteca scipy)")
    print("=" * 60)

    P_lu, L_lu, U_lu = la.lu(A_init)
    print("Matricea L (LU):")
    print(np.array2string(L_lu, precision=4, suppress_small=True))
    print("Matricea U (LU):")
    print(np.array2string(U_lu, precision=4, suppress_small=True))

    x_lib = np.linalg.solve(A_init, b)
    print("\nSolutia xlib (biblioteca):")
    print(np.array2string(x_lib, precision=8))

    # -------------------------------------------------------
    # 2. Descompunere LDLT (Cholesky) manuala + rezolvare
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. DESCOMPUNERE LDLT - CHOLESKY (implementare manuala)")
    print("=" * 60)

    try:
        x_chol, det_A, A_modified, d = solve_with_cholesky(A_init, b, eps)
    except ValueError as e:
        print(f"Eroare la descompunere: {e}")
        return

    print("Vectorul d (diagonala lui D):")
    print(np.array2string(d, precision=6))

    print("\nDeterminantul matricei A:")
    print(f"  det(A) = prod(d) = {det_A:.6e}")
    det_numpy = np.linalg.det(A_init)
    print(f"  det(A) numpy      = {det_numpy:.6e}  (verificare)")

    print("\nSolutia xChol (Cholesky manual):")
    print(np.array2string(x_chol, precision=8))

    # -------------------------------------------------------
    # 3. Verificare
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. VERIFICARE")
    print("=" * 60)

    # Calcul Ainit @ xChol manual (fara biblioteca, folosind A_modified)
    Ax = compute_Ainit_times_x(A_modified, d, x_chol)

    residual = Ax - b
    norm_residual = math.sqrt(sum(r ** 2 for r in residual))

    diff = x_chol - x_lib
    norm_diff = math.sqrt(sum(dv ** 2 for dv in diff))

    print(f"||A_init * xChol - b||_2  = {norm_residual:.2e}  (ar trebui < 1e-8)")
    print(f"||xChol - xlib||_2        = {norm_diff:.2e}  (ar trebui < 1e-9)")

    if norm_residual < 1e-8 and norm_diff < 1e-9:
        print("\n✓ Solutia calculata este corecta!")
    else:
        print("\n✗ Atentie: normele depasesc pragul asteptat.")

    # -------------------------------------------------------
    # Test pentru n > 100
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. TEST PENTRU n = 150")
    print("=" * 60)

    n_big = 150
    np.random.seed(0)
    A_big = generate_spd_matrix(n_big)
    b_big = np.random.rand(n_big)
    x_lib_big = np.linalg.solve(A_big, b_big)

    x_chol_big, det_big, A_mod_big, d_big = solve_with_cholesky(A_big, b_big, eps)

    Ax_big = compute_Ainit_times_x(A_mod_big, d_big, x_chol_big)
    res_big = math.sqrt(sum((Ax_big[i] - b_big[i]) ** 2 for i in range(n_big)))
    dif_big = math.sqrt(sum((x_chol_big[i] - x_lib_big[i]) ** 2 for i in range(n_big)))

    print(f"n = {n_big}")
    print(f"||A_init * xChol - b||_2  = {res_big:.2e}")
    print(f"||xChol - xlib||_2        = {dif_big:.2e}")

    if res_big < 1e-6 and dif_big < 1e-6:
        print("✓ Functioneaza corect si pentru n > 100!")


if __name__ == "__main__":
    main()