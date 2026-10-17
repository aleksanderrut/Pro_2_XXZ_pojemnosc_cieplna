using SparseArrays
using LinearAlgebra

# ==== GENEROWANIE BAZY ====
function gen_baza_xxz(L::Int; mag=:all)
    # Rozmiar bazy w zależności od trybu
    rozmiar_pelnej_bazy = 1 << L # przesówanie bitowe jedynki w lewo o L mijesc
    if magnetyzacja == :all
        liczba_stanow = rozmiar_pelnej_bazy
        sektory = 0:L #sektor zdefiniowany prz liczbę N_up w nim
    elseif magnetyzacja isa Real
        N_up = L/2 + mag # N_up = (N_up+N_down)/2 + (N_up - N_down)/2 
        N_up = Int(N_up) # tka dla pewność
        liczba_stanow = binomial(L, N_up)
        sektory = N_up:N_up #sektor zdefiniowany prz liczbę N_up w nim
    end
    # Tworzenie baz z uwzgledeniem sektorów magnetyzacji
    baza_int = zeros(Int, liczba_stanow)
    i = 1 # bo Julia
    for N_up_sektor in sektory
        for state in 0:(rozmiar_pelnej_bazy - 1)
            if count_ones(state) == N_up_sektor
                baza_int[i] = state
                i = i + 1
            end
        end
    end
    # Odwzorowanie odworotne stan -> indeks w bazie, zapisanw jako słownik
    indeks_odwrotny = Dict{Int, Int}() # typ słownik w juli
    for (i, state) in enumerate(baza_int)
        indeks_odwrotny[state] = i
    end
    return baza_int, indeks_odwrotny
end

# ==== GENEROWANIE HAMILTONIANU XXZ ====
function gen_ham_xxz(L::Int, J::Real, Delta::Real, PBC::Bool, baza_int::Vector{Int}, indeks_odwrotny::Dict{Int, Int})
    #tworzenie macierzy Hamiltonianu
    ile_stanow = length(baza_int)
    H = zeros(Float64, ile_stanow, ile_stanow)

    # Liczba połączeń między sąsiednimi spinami
    liczba_polaczen = L - 1
    if PBC == true && L > 2
        liczba_polaczen += 1
    end

    #pętla po stanch
    for i in 1:length(baza_int)
        state = baza_int[i]
        #pętla po możliwych przeskokach
        for s in 1:liczba_polaczen
            #struktura poąłczeń (dla PBC ostatni łączy się z pierwszym)
            if s == L
                t = 1
            else
                t = s + 1
            end
            #pozycje bitów bity numerujemy od najstarszych, czyli jakby od prawej do lewej
            bit_s = L - s
            bit_t = L - t
            #odczyt spinów z uzyciem bitowego and (bit&1) 0 = down, 1 = up
            spin_s = (state >> bit_s) & 1
            spin_t = (state >> bit_t) & 1

            # CZĘŚĆ DIAGONALNA
            if spin_s == spin_t
                H[i, i] += (J*Delta)/4
            else
                H[i, i] -= (J*Delta)/4
            end

            # CZĘŚĆ POZADIAGONALNA
            if spin_s != spin_t
                # Zamiana dwóch spinów przez XOR - ⊻ a to | - to or
                new_state = state ⊻ ((1 << bit_s) | (1 << bit_t))
                new_index = indeks_odwrotny[new_state] # Indeks nowego stanu w bazie
                H[new_index, i] += J / 2
            end
        end
    end
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