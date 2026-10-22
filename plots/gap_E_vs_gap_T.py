from pathlib import Path
import math
import re
import numpy as np
import matplotlib.pyplot as plt

FOLDER = Path(
    r"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_2_XXZ_pojemność_cieplna\data"
)

L_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
DELTA_WYBRANA = 0.5
J_WYBRANE = 1.0
MAG_WYBRANA = "all"       # "0", "1", "-1", "all"
PBC_WYBRANE = False

# Przełącznik normalizacji energii przez L
ENERGIA_NA_SPIN = False

# Przełączniki wyświetlania krzywych
WYSWIETL_DELTA_PARZYSTE = True
WYSWIETL_DELTA_NIEPARZYSTE = True
WYSWIETL_T_PARZYSTE = True
WYSWIETL_T_NIEPARZYSTE = True

# Tolerancja przy szukaniu pierwszego wyższego poziomu
TOLERANCJA_ENERGII = 1e-6

FOLDER_WYKRESY = FOLDER.parent / "plots"

# Nazwa np. eigenvalues_PBCtrue_L_x4_Delta0.5_J1.0_mag0.data
WZOR_NAZWY = re.compile(
    r"^eigenvalues_PBC(?P<PBC>true|false)_L_x(?P<L>\d+)"
    r"_Delta(?P<Delta>[-+0-9.eE]+)_J(?P<J>[-+0-9.eE]+)"
    r"_mag(?P<mag>all|[-+0-9.eE]+)\.data$"
)

# Nazwa np. Cv_max_PBCtrue_Delta0.5_J1.0_magall.data
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
    return math.isclose(float(mag_z_pliku), float(MAG_WYBRANA), abs_tol=1e-10)

# =============================================================================
def main():

    if not (
        WYSWIETL_DELTA_PARZYSTE
        or WYSWIETL_DELTA_NIEPARZYSTE
        or WYSWIETL_T_PARZYSTE
        or WYSWIETL_T_NIEPARZYSTE
    ):
        print("Nie wybrano żadnej krzywej.")
        return

    # Wyszukanie plików eigenvalues
    znalezione = {}
    for sciezka in FOLDER.glob("eigenvalues_*.data"):
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

    # Wyszukanie pliku Cv_max
    plik_Cv_max = None

    if WYSWIETL_T_PARZYSTE or WYSWIETL_T_NIEPARZYSTE:
        for sciezka in FOLDER.glob("Cv_max_*.data"):
            dopasowanie = WZOR_CV_MAX.fullmatch(sciezka.name)
            if dopasowanie is None:
                continue
            p = dopasowanie.groupdict()

            if (
                math.isclose(float(p["Delta"]), DELTA_WYBRANA, abs_tol=1e-10)
                and math.isclose(float(p["J"]), J_WYBRANE, abs_tol=1e-10)
                and (p["PBC"] == "true") == PBC_WYBRANE
                and mag_sie_zgadza(p["mag"])
            ):
                plik_Cv_max = sciezka
                break

    # Wczytanie tabeli: L, max_Cv, T_Cv_max
    T_max_dla_L = {}

    if WYSWIETL_T_PARZYSTE or WYSWIETL_T_NIEPARZYSTE:
        if plik_Cv_max is None:
            print("Nie znaleziono pliku Cv_max.")
        else:
            Cv_max_dane = np.atleast_2d(
                np.loadtxt(plik_Cv_max, dtype=float)
            )

            if Cv_max_dane.shape[1] != 3:
                raise ValueError("Plik Cv_max powinien mieć 3 kolumny.")

            T_max_dla_L = {
                int(wiersz[0]): wiersz[2]
                for wiersz in Cv_max_dane
            }

    # Oddzielne tablice dla czterech krzywych
    dane_delta_parzyste = []
    dane_delta_nieparzyste = []
    dane_T_parzyste = []
    dane_T_nieparzyste = []

    for L in L_VALUES:

        # Temperatura maksimum Cv
        if L in T_max_dla_L:
            T_Cv_max = T_max_dla_L[L]

            if L % 2 == 0 and WYSWIETL_T_PARZYSTE:
                dane_T_parzyste.append((1.0 / L, T_Cv_max))

            if L % 2 != 0 and WYSWIETL_T_NIEPARZYSTE:
                dane_T_nieparzyste.append((1.0 / L, T_Cv_max))

        # Przerwa energetyczna
        if L % 2 == 0 and not WYSWIETL_DELTA_PARZYSTE:
            continue

        if L % 2 != 0 and not WYSWIETL_DELTA_NIEPARZYSTE:
            continue

        if L not in znalezione:
            print(f"Brak pliku eigenvalues dla L = {L} — pomijam.")
            continue

        plik, _ = znalezione[L]
        E = np.sort(np.atleast_1d(np.loadtxt(plik, dtype=float)).ravel())

        if len(E) < 2:
            print(f"Widmo dla L = {L} ma mniej niż 2 energie — pomijam.")
            continue

        # Pierwsza energia większa od E0 o więcej niż 10^-6
        E0 = E[0]
        E_wyzsze = E[E - E0 > TOLERANCJA_ENERGII]

        if len(E_wyzsze) == 0:
            print(f"Brak wyższego poziomu energii dla L = {L}")
            continue

        E1 = E_wyzsze[0]

        # Normalizacja energii przez L
        if ENERGIA_NA_SPIN:
            E0 = E0 / L
            E1 = E1 / L

        # Przerwa energetyczna
        przerwa = E1 - E0

        # Podział na parzyste i nieparzyste L
        if L % 2 == 0:
            dane_delta_parzyste.append((1.0 / L, przerwa))
        else:
            dane_delta_nieparzyste.append((1.0 / L, przerwa))

        print(f"L={L:2d}, 1/L={1/L:.5f}, ΔE={przerwa:.8f}")

    # Sortowanie według 1/L
    dane_delta_parzyste.sort(key=lambda wiersz: wiersz[0])
    dane_delta_nieparzyste.sort(key=lambda wiersz: wiersz[0])
    dane_T_parzyste.sort(key=lambda wiersz: wiersz[0])
    dane_T_nieparzyste.sort(key=lambda wiersz: wiersz[0])

    # Opisy osi w zależności od normalizacji
    if ENERGIA_NA_SPIN:
        label_gap = r"$\Delta E=(E_1-E_0)/L$"
        nazwa_trybu = "na_spin"
    else:
        label_gap = r"$\Delta E=E_1-E_0$"
        nazwa_trybu = "calkowite"

    # Jeden wykres ze wspólną osią Y
    fig, ax = plt.subplots(figsize=(8, 5.3))

    nazwy_krzywych = []
    wszystkie_x = []

    # Delta dla parzystych L
    if WYSWIETL_DELTA_PARZYSTE and dane_delta_parzyste:
        dane = np.array(dane_delta_parzyste)
        x = dane[:, 0]
        y = dane[:, 1]

        ax.plot(
            x, y, "o-",
            color="tab:blue",
            label=label_gap + " (parzyste L)",
            linewidth=1.6,
            markersize=5
        )

        nazwy_krzywych.append("Delta_parzyste")
        wszystkie_x.extend(x)

    # Delta dla nieparzystych L
    if WYSWIETL_DELTA_NIEPARZYSTE and dane_delta_nieparzyste:
        dane = np.array(dane_delta_nieparzyste)
        x = dane[:, 0]
        y = dane[:, 1]

        ax.plot(
            x, y, "o-",
            color="tab:green",
            label=label_gap + " (nieparzyste L)",
            linewidth=1.6,
            markersize=5
        )

        nazwy_krzywych.append("Delta_nieparzyste")
        wszystkie_x.extend(x)

    # Temperatura maksimum Cv dla parzystych L
    if WYSWIETL_T_PARZYSTE and dane_T_parzyste:
        dane = np.array(dane_T_parzyste)
        x = dane[:, 0]
        y = dane[:, 1]

        ax.plot(
            x, y, "s-",
            color="tab:red",
            label=r"$T(C_V^{\max})$ (parzyste L)",
            linewidth=1.6,
            markersize=5
        )

        nazwy_krzywych.append("T_parzyste")
        wszystkie_x.extend(x)

    # Temperatura maksimum Cv dla nieparzystych L
    if WYSWIETL_T_NIEPARZYSTE and dane_T_nieparzyste:
        dane = np.array(dane_T_nieparzyste)
        x = dane[:, 0]
        y = dane[:, 1]

        ax.plot(
            x, y, "s-",
            color="tab:orange",
            label=r"$T(C_V^{\max})$ (nieparzyste L)",
            linewidth=1.6,
            markersize=5
        )

        nazwy_krzywych.append("T_nieparzyste")
        wszystkie_x.extend(x)

    if not wszystkie_x:
        print("Brak danych do wykresu.")
        plt.close(fig)
        return

    # Opis osi Y
    ma_delta = (
        "Delta_parzyste" in nazwy_krzywych
        or "Delta_nieparzyste" in nazwy_krzywych
    )

    ma_T = (
        "T_parzyste" in nazwy_krzywych
        or "T_nieparzyste" in nazwy_krzywych
    )

    ax.set_xlabel(r"$1/L$")

    if ma_delta and ma_T:
        ax.set_ylabel(r"$\Delta E,\ T(C_V^{\max})$")
    elif ma_delta:
        ax.set_ylabel(label_gap)
    else:
        ax.set_ylabel(r"$T(C_V^{\max})$")

    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=max(wszystkie_x) * 1.05)
    ax.legend(loc="best")

    # Parametry
    fig.suptitle(
        f"XXZ: Δ={DELTA_WYBRANA}, J={J_WYBRANE}, "
        f"mag={MAG_WYBRANA}, PBC={PBC_WYBRANE}",
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    # Zapis rysunku
    FOLDER_WYKRESY.mkdir(parents=True, exist_ok=True)

    nazwa_wykresu = "_".join(nazwy_krzywych)

    plik_wykresu = FOLDER_WYKRESY / (
        f"{nazwa_wykresu}_{nazwa_trybu}_Delta{DELTA_WYBRANA}_J{J_WYBRANE}"
        f"_mag{MAG_WYBRANA}_PBC{PBC_WYBRANE}.png"
    )

    fig.savefig(plik_wykresu, dpi=250, bbox_inches="tight")
    print(f"Zapisano wykres: {plik_wykresu}")

    plt.show()

if __name__ == "__main__":
    main()
