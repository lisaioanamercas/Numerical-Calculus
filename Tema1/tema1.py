import math

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

u_calculat = exercitiul_1()

print("-" * 50)

exercitiul_2_adunare(u_calculat)

print("-" * 50)

exercitiul_2_inmultire()