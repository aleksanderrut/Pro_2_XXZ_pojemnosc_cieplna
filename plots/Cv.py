from pathlib import Path
import math
import re
import numpy as np
import matplotlib.pyplot as plt

FOLDER = Path(
    r"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_2_XXZ_pojemność_cieplna\data"
)

# L_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
L_VALUES = [2, 4, 6, 8, 10, 12, 14]
DELTA_WYBRANA = 2.0
J_WYBRANE = 1.0
MAG_WYBRANA = "0"    # "0", "1", "-1", "all"
PBC_WYBRANE = False

# Zakres temperatur T/J
T_MIN = 0.0
T_MAX = 2.0

FOLDER_WYKRESY = FOLDER.parent / "plots"
WZOR_NAZWY = re.compile(
    r"^Cv_PBC(?P<PBC>true|false)_L_x(?P<L>\d+)"
    r"_Delta(?P<Delta>[-+0-9.eE]+)_J(?P<J>[-+0-9.eE]+)"
    r"_mag(?P<mag>all|[-+0-9.eE]+)\.data$"
)

def mag_sie_zgadza(mag_z_pliku):
    if str(MAG_WYBRANA) == "all":
        return mag_z_pliku == "all"

    if mag_z_pliku == "all":
        return False

    return math.isclose(float(mag_z_pliku), float(MAG_WYBRANA), abs_tol=1e-10)


# =============================================================================
def main():

    # Wyszukanie plików o odpowiednich parametrach
    znalezione = {}
    for sciezka in FOLDER.glob("Cv_*.data"):
        dopasowanie = WZOR_NAZWY.fullmatch(sciezka.name)
        if dopasowanie is None:
            continue
        p = dopasowanie.groupdict()
        L = int(p["L"])

        if (
            L in L_VALUES
            and math.isclose(float(p["Delta"]), DELTA_WYBRANA, abs_tol=1e-10)
            and math.isclose(float(p["J"]), J_WYBRANE, abs_tol=1e-10)
            and (p["PBC"] == "true") == PBC_WYBRANE
            and mag_sie_zgadza(p["mag"])
        ):
            znalezione[L] = (sciezka, p)

    # =============================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    liczba_krzywych = 0
    parametry = None
    for L in L_VALUES:
        if L not in znalezione:
            print(f"Brak pliku dla L = {L} — pomijam.")
            continue

        plik, parametry = znalezione[L]
        # Wczytanie danych: T, <E>, <E²>, Cv
        dane = np.atleast_2d(np.loadtxt(plik))

        # Temperatura i pojemność cieplna
        T = dane[:, 0]
        Cv = dane[:, 3]

        # Normalizacja
        T_J = T / J_WYBRANE
        Cv_na_L = Cv / L

        # Wybrany zakres temperatur
        maska = (T_J >= T_MIN) & (T_J <= T_MAX)
        T_J = T_J[maska]
        Cv_na_L = Cv_na_L[maska]

        # Rysowanie krzywej
        ax.plot(T_J, Cv_na_L, linewidth=1.8, label=fr"$L={L}$")
        liczba_krzywych += 1
        print(f"L = {L}, źródło: {plik.name}")

    # =============================================================================
    ax.set_xlabel(r"$T/J$", fontsize=12)
    ax.set_ylabel(r"$C_V/L$", fontsize=12)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10)

    ax.set_title(
        f"XXZ: Δ = {parametry['Delta']}, J = {parametry['J']}, "
        f"mag = {parametry['mag']}, PBC = {parametry['PBC']}",
        fontsize=12,
        fontweight="normal"
    )
    fig.tight_layout()

    # =============================================================================
    FOLDER_WYKRESY.mkdir(parents=True, exist_ok=True)
    nazwa = (
        f"Cv_na_L_Delta{parametry['Delta']}_J{parametry['J']}"
        f"_mag{parametry['mag']}_PBC{parametry['PBC']}.png"
    )
    plik_wykresu = FOLDER_WYKRESY / nazwa
    fig.savefig(plik_wykresu, dpi=300, bbox_inches="tight")
    print(f"Zapisano wykres: {plik_wykresu}")

    plt.show()

if __name__ == "__main__":
    main()
