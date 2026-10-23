
from pathlib import Path
import math
import re
import numpy as np
import matplotlib.pyplot as plt

FOLDER = Path(
    r"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_2_XXZ_pojemność_cieplna\data"
)

# PARAMETRY
L_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
DELTA_WYBRANA = 0.5
J_WYBRANE = 1.0
MAG_WYBRANA = "all"
PBC_WYBRANE = False

# Przełączniki krzywych
WYSWIETL_PARZYSTE = True
WYSWIETL_NIEPARZYSTE = True

# Tolerancja energii
TOLERANCJA_ENERGII = 1e-6

FOLDER_WYKRESY = FOLDER.parent / "plots"

# Wzór nazwy plików z wartościami własnymi
WZOR_NAZWY = re.compile(
    r"^eigenvalues_PBC(?P<PBC>true|false)_L_x(?P<L>\d+)"
    r"_Delta(?P<Delta>[-+0-9.eE]+)_J(?P<J>[-+0-9.eE]+)"
    r"_mag(?P<mag>all|[-+0-9.eE]+)\.data$"
)

# Wzór nazwy pliku z maksimum Cv
WZOR_CV_MAX = re.compile(
    r"^Cv_max_PBC(?P<PBC>true|false)"
    r"_Delta(?P<Delta>[-+0-9.eE]+)_J(?P<J>[-+0-9.eE]+)"
    r"_mag(?P<mag>all|[-+0-9.eE]+)\.data$"
)


def mag_sie_zgadza(mag_z_pliku):
    if str(MAG_WYBRANA) == "all":
        return mag_z_pliku == "all"

    if mag_z_pliku == "all":
        return False

    return math.isclose(
        float(mag_z_pliku),
        float(MAG_WYBRANA),
        abs_tol=1e-10
    )


def parametry_sie_zgadzaja(p):
    return (
        math.isclose(float(p["Delta"]), DELTA_WYBRANA, abs_tol=1e-10)
        and math.isclose(float(p["J"]), J_WYBRANE, abs_tol=1e-10)
        and (p["PBC"] == "true") == PBC_WYBRANE
        and mag_sie_zgadza(p["mag"])
    )


def main():

    if not WYSWIETL_PARZYSTE and not WYSWIETL_NIEPARZYSTE:
        print("Nie wybrano żadnej krzywej.")
        return

    # ============================================================
    # 1. WYSZUKIWANIE PLIKÓW EIGENVALUES
    # ============================================================

    znalezione = {}

    for sciezka in FOLDER.glob("eigenvalues_*.data"):

        dopasowanie = WZOR_NAZWY.fullmatch(sciezka.name)

        if dopasowanie is None:
            continue

        p = dopasowanie.groupdict()
        L = int(p["L"])

        if L in L_VALUES and parametry_sie_zgadzaja(p):
            znalezione[L] = sciezka

    # ============================================================
    # 2. WCZYTANIE TEMPERATUR MAKSIMUM Cv
    # ============================================================

    plik_Cv_max = None

    for sciezka in FOLDER.glob("Cv_max_*.data"):

        dopasowanie = WZOR_CV_MAX.fullmatch(sciezka.name)

        if dopasowanie is None:
            continue

        p = dopasowanie.groupdict()

        if parametry_sie_zgadzaja(p):
            plik_Cv_max = sciezka
            break

    if plik_Cv_max is None:
        print("Nie znaleziono pliku Cv_max.")
        return

    # Kolumny: L, Cv_max, T(Cv_max)
    Cv_max_dane = np.atleast_2d(
        np.loadtxt(plik_Cv_max, dtype=float)
    )

    if Cv_max_dane.shape[1] != 3:
        raise ValueError("Plik Cv_max musi mieć 3 kolumny.")

    T_max_dla_L = {
        int(wiersz[0]): wiersz[2]
        for wiersz in Cv_max_dane
    }

    # ============================================================
    # 3. OBLICZENIE DELTA E / T
    # ============================================================

    dane_parzyste = []
    dane_nieparzyste = []

    for L in L_VALUES:

        if L % 2 == 0 and not WYSWIETL_PARZYSTE:
            continue

        if L % 2 != 0 and not WYSWIETL_NIEPARZYSTE:
            continue

        if L not in znalezione:
            print(f"Brak eigenvalues dla L = {L}")
            continue

        if L not in T_max_dla_L:
            print(f"Brak temperatury maksimum Cv dla L = {L}")
            continue

        # Wczytanie i sortowanie energii
        E = np.sort(
            np.atleast_1d(
                np.loadtxt(znalezione[L], dtype=float)
            ).ravel()
        )

        if len(E) < 2:
            print(f"Za mało poziomów energii dla L = {L}")
            continue

        # Energia stanu podstawowego
        E0 = E[0]

        # Pierwszy poziom wyższy od E0 z tolerancją
        E_wyzsze = E[E - E0 > TOLERANCJA_ENERGII]

        if len(E_wyzsze) == 0:
            print(f"Brak wyższego poziomu dla L = {L}")
            continue

        E1 = E_wyzsze[0]

        # Przerwa energetyczna
        delta_E = E1 - E0

        # Temperatura maksimum pojemności cieplnej
        T_max = T_max_dla_L[L]

        if not np.isfinite(T_max) or T_max <= 0:
            print(f"Niepoprawna temperatura dla L = {L}")
            continue

        # Stosunek przerwy energetycznej do temperatury
        stosunek = T_max / delta_E 

        # Podział na parzyste i nieparzyste L
        if L % 2 == 0:
            dane_parzyste.append((L, stosunek))
        else:
            dane_nieparzyste.append((L, stosunek))

        print(
            f"L={L:2d}, "
            f"Delta E={delta_E:.8f}, "
            f"T_max={T_max:.8f}, "
            f"Delta E/T={stosunek:.8f}"
        )

    # ============================================================
    # 4. RYSOWANIE WYKRESU
    # ============================================================

    if not dane_parzyste and not dane_nieparzyste:
        print("Brak danych do wykresu.")
        return

    fig, ax = plt.subplots(figsize=(8, 5.3))

    if WYSWIETL_PARZYSTE and dane_parzyste:
        dane = np.array(dane_parzyste)

        ax.plot(
            dane[:, 0],
            dane[:, 1],
            "o-",
            color="tab:blue",
            label="Parzyste L",
            linewidth=1.6,
            markersize=6
        )

    if WYSWIETL_NIEPARZYSTE and dane_nieparzyste:
        dane = np.array(dane_nieparzyste)

        ax.plot(
            dane[:, 0],
            dane[:, 1],
            "s--",
            color="tab:orange",
            label="Nieparzyste L",
            linewidth=1.6,
            markersize=6
        )

    # Opisy osi
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$T(C_V^{\max} / \Delta E$")

    ax.set_xticks(L_VALUES)
    ax.set_xlim(min(L_VALUES) - 0.5, max(L_VALUES) + 0.5)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Tytuł zawierający parametry modelu XXZ
    ax.set_title(
        rf"XXZ: $\Delta={DELTA_WYBRANA}$, $J={J_WYBRANE}$, "
        rf"mag={MAG_WYBRANA}, PBC={PBC_WYBRANE}",
        fontsize=13,
        pad=12
    )

    fig.tight_layout()

    # ============================================================
    # 5. ZAPIS WYKRESU
    # ============================================================

    FOLDER_WYKRESY.mkdir(parents=True, exist_ok=True)

    plik_wykresu = FOLDER_WYKRESY / (
        f"DeltaE_przez_T_Delta{DELTA_WYBRANA}_J{J_WYBRANE}"
        f"_mag{MAG_WYBRANA}_PBC{PBC_WYBRANE}.png"
    )

    fig.savefig(
        plik_wykresu,
        dpi=250,
        bbox_inches="tight"
    )

    print(f"Zapisano wykres: {plik_wykresu}")

    plt.show()


if __name__ == "__main__":
    main()
