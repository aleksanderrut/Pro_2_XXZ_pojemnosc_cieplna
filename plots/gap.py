from pathlib import Path
import math
import re
import numpy as np
import matplotlib.pyplot as plt

FOLDER = Path(
    r"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_2_XXZ_pojemność_cieplna\data"
)

L_VALUES = [2, 4, 6, 8, 10, 12]
DELTA_WYBRANA = 2.0
J_WYBRANE = 1.0
MAG_WYBRANA = "all"       # "0", "1", "-1", "all"
PBC_WYBRANE = True

# Przełącznik normalizacji energii przez L
ENERGIA_NA_SPIN = True

FOLDER_WYKRESY = FOLDER.parent / "plots"

# Nazwa np. eigenvalues_PBCtrue_L_x4_Delta0.5_J1.0_mag0.data
WZOR_NAZWY = re.compile(
    r"^eigenvalues_PBC(?P<PBC>true|false)_L_x(?P<L>\d+)"
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

    # Zbieranie E0, E1 i przerwy dla kolejnych L
    dane = []

    for L in L_VALUES:
        if L not in znalezione:
            print(f"Brak pliku dla L = {L} — pomijam.")
            continue
        plik, _ = znalezione[L]
        E = np.sort(np.atleast_1d(np.loadtxt(plik, dtype=float)).ravel())
        if len(E) < 2:
            print(f"Widmo dla L = {L} ma mniej niż 2 energie — pomijam.")
            continue
        E0 = E[0]
        E1 = E[1]
        # Normalizacja energii przez L
        if ENERGIA_NA_SPIN:
            E0 = E0 / L
            E1 = E1 / L
        # Przerwa wyznaczana z wybranych energii
        przerwa = E1 - E0
        dane.append((1.0 / L, E0, E1, przerwa))
        print(f"L={L:2d}, 1/L={1/L:.5f}, E0={E0:.8f}, E1={E1:.8f}, ΔE={przerwa:.8f}")

    # Sortowanie według 1/L
    dane = np.array(sorted(dane, key=lambda wiersz: wiersz[0]))

    x = dane[:, 0]
    E0_values = dane[:, 1]
    E1_values = dane[:, 2]
    gap_values = dane[:, 3]

    # Opisy osi w zależności od normalizacji
    if ENERGIA_NA_SPIN:
        label_E0 = r"$E_0/L$"
        label_E1 = r"$E_1/L$"
        label_gap = r"$ΔE=(E_1-E_0)/L$"
        label_y = r"Energia na spin $E/L$"
        nazwa_trybu = "na_spin"
    else:
        label_E0 = r"$E_0$"
        label_E1 = r"$E_1$"
        label_gap = r"$ΔE=E_1-E_0$"
        label_y = r"Energia $E$"
        nazwa_trybu = "calkowite"

    # Dwa panele
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))

    # Panel 1: dwa najniższe poziomy energii
    ax1.plot(x, E0_values, "o-", label=label_E0, linewidth=1.6, markersize=5)
    ax1.plot(x, E1_values, "s-", label=label_E1, linewidth=1.6, markersize=5)
    ax1.set_xlabel(r"$1/L$")
    ax1.set_ylabel(label_y)
    ax1.set_title("Dwa najniższe poziomy energii")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: przerwa energetyczna
    ax2.plot(x, gap_values, "o-", label=label_gap,linewidth=1.6, markersize=5)
    ax2.set_xlabel(r"$1/L$")
    ax2.set_ylabel(label_gap)
    ax2.set_title("Przerwa energetyczna")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    for ax in (ax1, ax2):
        ax.set_xlim(left=0, right=max(x) * 1.05)

    # Parametry odczytane z nazwy znalezionego pliku
    parametry = znalezione[next(L for L in L_VALUES if L in znalezione)][1]

    fig.suptitle(
        f"XXZ: Δ={parametry['Delta']}, J={parametry['J']}, "
        f"mag={parametry['mag']}, PBC={parametry['PBC']}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    # Zapis rysunku
    FOLDER_WYKRESY.mkdir(parents=True, exist_ok=True)
    plik_wykresu = FOLDER_WYKRESY / (
        f"energie_przerwa_{nazwa_trybu}_Delta{parametry['Delta']}_J{parametry['J']}"
        f"_mag{parametry['mag']}_PBC{parametry['PBC']}.png"
    )
    fig.savefig(plik_wykresu, dpi=250, bbox_inches="tight")
    print(f"Zapisano wykres: {plik_wykresu}")

    plt.show()

if __name__ == "__main__":
    main()
