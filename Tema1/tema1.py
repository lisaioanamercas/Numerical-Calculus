import math
import random
from time import time

def exercitiul_1():
    print("---  Gasire precizie masina (u) ---")
    u = 1.0
    m = 0
    
    # Folosim u/10 inauntru pentru a gasi pragul unde adunarea "esueaza"
    while (1.0 + (u / 10.0)) != 1.0:
        m += 1
        u = 10.0**(-m)
    
    print(f"Cel mai micutz u de forma 10^-m gasit este: 10^-{m}")
    print(f"Valoarea exacta a lui u: {u:.20e}")
    return u

def exercitiul_2_adunare(u):
    print("\n--- Exercitiul 2: Neasociativitatea adunarii ---")
    x = 1.0
    y = u / 10.0
    z = u / 10.0
    
    rezultat_stanga = (x + y) + z
    
    # Aici (10^-16 + 10^-16) se face intai si da 2e-16, care e destul de mare 
    # ca sa schimbe valoarea lui 1.0 cand sunt adunate
    rezultat_dreapta = x + (y + z)
    
    print(f"Calcul (x + y) + z = {rezultat_stanga:.20f}")
    print(f"Calcul x + (y + z) = {rezultat_dreapta:.20f}")
    
    if rezultat_stanga != rezultat_dreapta:
        print("Rezultat: Operatia de adunare este neasociativa!")
    else:
        print("Rezultat: Operatiile par egale (verifica precizia).")

def exercitiul_2_inmultire():
    print("\n--- Exercitiul 2: Neasociativitatea inmultirii ---")
    
    x = 1e308  # Aproape de valoarea maxima reprezentabila Z [cite: 547]
    y = 1.1
    z = 0.5
    
    # In prima varianta, x * y depaseste limita si devine 'inf' (infinity) 
    rez_stanga = (x * y) * z
    
    # In a doua varianta, y * z se face intai (0.55), iar x * 0.55 ramane un numar finit
    rez_dreapta = x * (y * z)
    
    print(f"Calcul (x * y) * z = {rez_stanga}")
    print(f"Calcul x * (y * z) = {rez_dreapta}")
    
    if rez_stanga != rez_dreapta:
        print("Rezultat: Operatia de inmultire este neasociativa! ")

# ---------------------------------------------------------------------------
# Reducerea argumentului la intervalul (-pi/2, pi/2) folosind periodicitatea
# si la (-pi/4, pi/4) prin antisimetrie; valori multiple de pi/2 tratate separat
# ---------------------------------------------------------------------------

def _reduce_argument(x):
    """Reduce x la intervalul (-pi/2, pi/2) folosind periodicitatea tan."""
    y = math.remainder(x, math.pi)  # y in (-pi/2, pi/2]
    half_pi = math.pi / 2.0
    tol = 1e-15

    # Verifica daca x este multiplu de pi/2 (tan nedefinit)
    k = round(y / half_pi)
    if k % 2 != 0 and abs(y - k * half_pi) < tol:
        return None, math.copysign(math.inf, math.sin(y))

    return y, None


# ---------------------------------------------------------------------------
# Metoda 1: Fractii continue – algoritmul Lentz modificat
#
# Reprezentarea tan ca fractie continua:
#   tan(x) = x / (1 + (-x^2) / (3 + (-x^2) / (5 + ...)))
#
# Adica: b0=0, a1=x, b1=1, iar pentru j>=2: a_j = -x^2, b_j = 2j-1
# Conform PDF: f0 = b0 (=0 => inlocuit cu mic), C0 = f0, D0 = 0
# ---------------------------------------------------------------------------

def tan_continua(x, epsilon=1e-12, max_iter=10000):
    """Aproximeaza tan(x) folosind fractii continue (Lentz modificat).

    Parametri
    ---------
    x       : argumentul
    epsilon : precizia dorita (criteriu de oprire |delta-1| < epsilon)
    """
    y, special = _reduce_argument(x)
    if special is not None:
        return special

    # Antisimetrie: tan(-x) = -tan(x)
    if y < 0.0:
        return -tan_continua(-y, epsilon=epsilon, max_iter=max_iter)

    tiny = 1e-12

    # Initializare conform algoritmului Lentz
    # b0 = 0 pentru tan (fractia incepe cu x/1+...)
    # Folosim reprezentarea echivalenta unde b0=0, a1=x, b1=1
    # In practica tratam primul pas manual pentru a evita impartirea la 0.
    #
    # f0 = b0 = 0 => f0 = tiny (evitam zero)
    f = tiny
    C = tiny   # C0 = f0
    D = 0.0    # D0 = 0

    for j in range(1, max_iter + 1):
        # Coeficientii fractiei continue pentru tan:
        #   j=1 : a1 = x,    b1 = 1
        #   j>=2: aj = -x^2, bj = 2j-1
        if j == 1:
            a_j = y
        else:
            a_j = -(y * y)

        b_j = float(2 * j - 1)

        # Actualizare D (= 1 / (b_j + a_j * D_{j-1}))
        D = b_j + a_j * D
        if D == 0.0:
            D = tiny
        D = 1.0 / D

        # Actualizare C (= b_j + a_j / C_{j-1})
        C = b_j + a_j / C
        if C == 0.0:
            C = tiny

        delta = C * D
        f *= delta

        if abs(delta - 1.0) < epsilon:
            return f

    # Daca nu a convergentt, returnam cea mai buna aproximare
    return f


# ---------------------------------------------------------------------------
# Metoda 2: Aproximare polinomiala (Maclaurin / Horner)
#
# tan(x) ≈ x + (1/3)x^3 + (2/15)x^5 + (17/315)x^7 + (62/2835)x^9
#         = x + x^3 * (c1 + x^2*(c2 + x^2*(c3 + x^2*c4)))
#
# Valabila doar pentru x in (-pi/4, pi/4).
# Pentru x in [pi/4, pi/2) se foloseste: tan(x) = 1 / tan(pi/2 - x)
# ---------------------------------------------------------------------------

def _tan_polinom_baza(x):
    """Evalueaza polinomul Maclaurin pentru x in (-pi/4, pi/4) via Horner."""
    c1 = 0.33333333333333333   # 1/3
    c2 = 0.13333333333333333   # 2/15
    c3 = 0.053968253968254     # 17/315
    c4 = 0.0218694885361552    # 62/2835

    x2 = x * x
    x3 = x2 * x
    # Schema Horner: x + x3*(c1 + x2*(c2 + x2*(c3 + x2*c4)))
    return x + x3 * (c1 + x2 * (c2 + x2 * (c3 + x2 * c4)))


def tan_polinomial(x):
    """Aproximeaza tan(x) folosind polinomul Maclaurin."""
    y, special = _reduce_argument(x)
    if special is not None:
        return special

    if y < 0.0:
        return -tan_polinomial(-y)

    quarter_pi = math.pi / 4.0
    half_pi = math.pi / 2.0

    if abs(y) <= quarter_pi:
        return _tan_polinom_baza(y)
    else:
        # tan(x) = 1 / tan(pi/2 - x) pentru x in [pi/4, pi/2)
        return 1.0 / _tan_polinom_baza(half_pi - y)


def exercitiul_3(epsilon):
    """Compara cele doua metode de aproximare a tangentei.

    Parametri
    ---------
    epsilon : precizia pentru metoda fractiilor continue (citit de la tastatura)
    """
    print("\n--- Exercitiul 3: Aproximarea functiei tangenta ---")

    numar_valori = 10000
    seed = 42
    margine = 1e-10

    rng = random.Random(seed)
    valori = [
        rng.uniform(-math.pi / 2.0 + margine, math.pi / 2.0 - margine)
        for _ in range(numar_valori)
    ]

    # Metoda 1: Fractii continue
    start_cf = time.perf_counter()
    rezultate_cf = [tan_continua(x, epsilon=epsilon) for x in valori]
    timp_cf = time.perf_counter() - start_cf

    # Metoda 2: Polinomiala
    start_poly = time.perf_counter()
    rezultate_poly = [tan_polinomial(x) for x in valori]
    timp_poly = time.perf_counter() - start_poly

    # Valori de referinta (biblioteca math)
    valori_ref = [math.tan(x) for x in valori]

    erori_cf   = [abs(valori_ref[i] - rezultate_cf[i])   for i in range(numar_valori)]
    erori_poly = [abs(valori_ref[i] - rezultate_poly[i]) for i in range(numar_valori)]

    print(f"Numar valori testate : {numar_valori}")
    print(f"Epsilon (fractii continue): {epsilon:.2e}")

    print("\nMetoda fractii continue (Lentz modificat):")
    print(f"  Eroare maxima : {max(erori_cf):.6e}")
    print(f"  Eroare medie  : {sum(erori_cf) / numar_valori:.6e}")
    print(f"  Timp total    : {timp_cf:.6f} sec")

    print("\nMetoda polinomiala (Maclaurin):")
    print(f"  Eroare maxima : {max(erori_poly):.6e}")
    print(f"  Eroare medie  : {sum(erori_poly) / numar_valori:.6e}")
    print(f"  Timp total    : {timp_poly:.6f} sec")


def main():
    u = exercitiul_1()

    print("-" * 50)

    exercitiul_2_adunare(u)

    print("-" * 50)

    exercitiul_2_inmultire()

    print("-" * 50)

    # Citire epsilon de la tastatura (cerinta PDF: parametru de intrare)
    while True:
        try:
            epsilon_str = input(
                "\nIntroduceti precizia epsilon pentru metoda fractiilor continue "
                "(ex: 1e-12): "
            ).strip()
            epsilon = float(epsilon_str)
            if epsilon <= 0:
                raise ValueError("Epsilon trebuie sa fie pozitiv.")
            break
        except ValueError as e:
            print(f"Valoare invalida: {e}. Incercati din nou.")

    exercitiul_3(epsilon)

    print("-" * 50)
    
if __name__ == "__main__":
    main()