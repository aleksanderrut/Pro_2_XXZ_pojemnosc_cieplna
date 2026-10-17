using SparseArrays
using LinearAlgebra

# ==== GENEROWANIE BAZY XXZ ====
function gen_baza_xxz(L::Int, N_up::Int; sector_mode::Symbol=:single)
    # Rozmiar bazy w zależności od trybu
    rozmiar_pelnej_bazy = 1 << L
    if sector_mode == :single
        liczba_stanow = binomial(L, N_up)
        sektory = N_up:N_up
    elseif sector_mode == :all
        liczba_stanow = rozmiar_pelnej_bazy
        sektory = 0:L
    end

    baza_int = zeros(Int, liczba_stanow)

    i = 1

    for N_up_sektor in sektory
        for state in 0:(rozmiar_pelnej_bazy - 1)

            if count_ones(state) == N_up_sektor
                baza_int[i] = state
                i += 1
            end

        end
    end

    # Odwzorowanie: stan -> indeks w bazie
    indeks_odwrotny = Dict{Int, Int}()

    for (i, state) in enumerate(baza_int)
        indeks_odwrotny[state] = i
    end

    return baza_int, indeks_odwrotny
end


# ============================================================
# GENEROWANIE HAMILTONIANU XXZ
# ============================================================

function gen_ham_xxz(M::Int, J::Real, Delta::Real, pbc::Bool,
                     baza_int::Vector{Int},
                     indeks_odwrotny::Dict{Int, Int})

    ile_stanow = length(baza_int)

    # Tablice do konstrukcji macierzy rzadkiej
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # Lista par sąsiednich spinów
    bonds = [(s, s + 1) for s in 1:(M - 1)]

    # Dodajemy połączenie okresowe
    if pbc && M > 2
        push!(bonds, (M, 1))
    end


    # ========================================================
    # KONSTRUKCJA HAMILTONIANU
    # ========================================================

    for (i, state) in enumerate(baza_int)

        diagonal = 0.0

        for (s, t) in bonds

            # Pozycje bitów odpowiadających spinom s oraz t
            bit_s = M - s
            bit_t = M - t

            # Odczyt spinów: 0 = down, 1 = up
            spin_s = (state >> bit_s) & 1
            spin_t = (state >> bit_t) & 1


            # ------------------------------------------------
            # CZĘŚĆ DIAGONALNA
            # J * Delta * Sz_s * Sz_t
            # ------------------------------------------------

            if spin_s == spin_t
                diagonal += J * Delta / 4
            else
                diagonal -= J * Delta / 4
            end


            # ------------------------------------------------
            # CZĘŚĆ POZADIAGONALNA
            # J/2 * (S+_s S-_t + S-_s S+_t)
            # ------------------------------------------------

            if spin_s != spin_t

                # Maska wskazująca dwa spiny
                mask = (1 << bit_s) | (1 << bit_t)

                # Zamiana spinów przez XOR
                new_state = state ⊻ mask

                # Indeks nowego stanu w bazie
                new_index = indeks_odwrotny[new_state]

                push!(rows, new_index)
                push!(cols, i)
                push!(vals, Float64(J / 2))
            end
        end

        # Dodanie elementu diagonalnego
        push!(rows, i)
        push!(cols, i)
        push!(vals, diagonal)
    end

    H = sparse(rows, cols, vals, ile_stanow, ile_stanow)

    return H
end


function main()
    L = 6
    N_up = 3
    J = 1.0
    Delta = 0.5
    pbc = true

    # :single -> jeden sektor Sz
    # :all    -> wszystkie sektory Sz
    sector_mode = :single

    baza_int, indeks_odwrotny = gen_baza_xxz(M, N_up; sector_mode=sector_mode)
    H = gen_ham_xxz(M, J, Delta, pbc, baza_int, indeks_odwrotny)

end


main()