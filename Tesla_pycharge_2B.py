"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""



"""
Tesla_pycharge_2B.py  —  GRUPA 2, varijanta 2B (pycharge motor)

Ista struktura kao GRUPA 1 i 2A:
  motor (EM polje/potencijal pokretnog naboja)  ->  primena na 4630 izvlacenja  ->  skor  ->  rangirane kombinacije

pycharge uvodi pravi EM izvor: oscilujuci tackasti naboj.
Izracunavam skalarni potencijal duz pravca prostiranja:
  S(x)   = skalarni potencijal duz x-linije
  E_x    = -dS/dx  (uzduzna komponenta iz potencijala)
"""


import numpy as np

from Tesla_Scalar_1 import (
    SEED,
    W_TALAS,
    W_FREQ,
    CSV_PATH,
    MIN_BROJ,
    MAX_BROJ,
    OUTPUT_DIR,
    ucitaj_izvlacenja,
    glavne_mere,
    ne_frekvencijski_skor,
    frekvencija_brojeva,
    kombinovani_skor,
    izaberi_kombinacije,
    skor_kombinacije,
    nacrtaj_polje,
)

OSNOVA = "tesla_pycharge_2B"


def _normalizuj_signal(S):
    """Vrati signal centriran i skaliran na stabilan opseg."""
    S = np.asarray(S, dtype=float)
    S = S - np.mean(S)
    m = np.max(np.abs(S))
    if m <= 0:
        return S
    return S / m


def simuliraj_pycharge(nx=4630, broj_tacaka=512):
    """pycharge motor. Vrati (x, S, E_x) duz pravca prostiranja, duzine nx.

    Model: jedan oscilujuci tackasti naboj, a S(x) je njegov skalarni potencijal
    meren po x-liniji sa malim y-pomerajem da ne pogodim singularitet naboja.
    """
    import jax
    import jax.numpy as jnp
    from pycharge import Charge, potentials_and_fields

    q = 1.602176634e-19
    amplituda = 2.0e-9
    omega = 7.5e16
    y_pomeraj = 0.5e-9
    lim = 8.0e-9
    t0 = 2.0e-16

    naboj = Charge(lambda t: [amplituda * jnp.sin(omega * t), 0.0, 0.0], q)
    kolicine_fn = jax.jit(potentials_and_fields([naboj]))

    x_raw = jnp.linspace(-lim, lim, broj_tacaka)
    X = x_raw[:, None, None, None]
    Y = jnp.full_like(X, y_pomeraj)
    Z = jnp.zeros_like(X)
    T = jnp.full_like(X, t0)

    kolicine = kolicine_fn(X, Y, Z, T)
    S_raw = np.asarray(kolicine.scalar[:, 0, 0, 0])
    S_raw = _normalizuj_signal(S_raw)

    x = np.linspace(0.0, 1.0, nx)
    S = np.interp(x, np.linspace(0.0, 1.0, broj_tacaka), S_raw)
    E_x = -np.gradient(S, x[1] - x[0])
    return x, S, E_x


def main():
    # --- Korak 1: motor (pycharge) ---
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)
    x, S, E_x = simuliraj_pycharge(nx=n)
    mere = glavne_mere(S, E_x)
    print()
    print("Tesla Scalar / GRUPA 2 - 2B (pycharge motor)")
    print("Talas: EM skalarni potencijal oscilujuceg naboja duz x-pravca")
    print("Uzduzno polje: E_x = -dS/dx")
    print()
    print(f"broj tacaka: {len(x)}")
    print(f"max S: {mere['max_S']:.10f}")
    print(f"max |E_x|: {mere['max_abs_E_x']:.10f}")
    print(f"ukupna gustina energije: {mere['ukupna_gustina_energije']:.10f}")
    print()

    # --- Korak 2: primena talasa na CSV + prava frekvencija ---
    energija = 0.5 * (S ** 2 + E_x ** 2)
    talas_skor, _ = ne_frekvencijski_skor(izvlacenja, energija)
    udeo, pojave = frekvencija_brojeva(izvlacenja)
    skor = kombinovani_skor(talas_skor, udeo)
    poredak = sorted(skor.items(), key=lambda kv: kv[1], reverse=True)
    freq_poredak = sorted(pojave, key=lambda b: (pojave[b], b), reverse=True)
    kombinacije = izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED)
    rangirane_kombinacije = sorted(
        ((k, skor_kombinacije(k, skor)) for k in kombinacije),
        key=lambda kv: kv[1],
        reverse=True,
    )
    png, jpg = nacrtaj_polje(x, S, E_x, osnova=OSNOVA)

    with open(OUTPUT_DIR / f"{OSNOVA}.txt", "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - GRUPA 2 / 2B (pycharge: talas + prava frekvencija)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED} | tezine: talas={W_TALAS} freq={W_FREQ}\n\n")
        f.write("Brojevi po kombinovanom skoru (tezinski talas + frekvencija):\n")
        for b, s in poredak:
            f.write(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})\n")

        f.write("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):\n")
        f.write("  broj | pojava |   udeo\n")
        f.write("  -----+--------+--------\n")
        for b in freq_poredak:
            f.write(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}\n")
        f.write(f"  ukupno pojava: {sum(pojave.values())}\n")

        f.write("\nPredlozene kombinacije (rangirane po skoru kombinacije):\n")
        for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
            f.write(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}\n")

        f.write("\nSlike talasa/polja:\n")
        f.write(f"  PNG: {png}\n")
        f.write(f"  JPG: {jpg}\n")

    print()
    print("\nTesla Scalar - GRUPA 2 / 2B (pycharge: talas + prava frekvencija)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n} | tezine: talas={W_TALAS} freq={W_FREQ}")
    print("\nTop 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):")
    for b, s in poredak[:10]:
        print(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})")

    print()
    print("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):")
    print("  broj | pojava |   udeo")
    print("  -----+--------+--------")
    for b in freq_poredak:
        print(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}")
    print(f"  ukupno pojava: {sum(pojave.values())}")

    print()
    print("\nPredlozene kombinacije (rangirane po skoru kombinacije):")
    for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
        print(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}")
    print(f"\nSacuvano: {OUTPUT_DIR / f'{OSNOVA}.txt'}")
    print()


if __name__ == "__main__":
    main()



"""
Tesla Scalar / GRUPA 2 - 2B (pycharge motor)
Talas: EM skalarni potencijal oscilujuceg naboja duz x-pravca
Uzduzno polje: E_x = -dS/dx

broj tacaka: 4630
max S: 0.9999911509
max |E_x|: 17.6018081226
ukupna gustina energije: 50591.8632273473

Slika talasa: /Tesla/tesla_pycharge_2B.png
Slika talasa: /Tesla/tesla_pycharge_2B.jpg


Tesla Scalar - GRUPA 2 / 2B (pycharge: talas + prava frekvencija)
CSV: /data/loto7hh_4630_k46.csv | Izvlacenja: 4630 | tezine: talas=0.7 freq=0.3

Top 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):
  22  skor=0.8470146085  freq=0.02626  (pojava=851)
  31  skor=0.8333333333  freq=0.02561  (pojava=830)
  03  skor=0.7294195195  freq=0.02546  (pojava=825)
  37  skor=0.7070035034  freq=0.02654  (pojava=860)
  14  skor=0.6745677823  freq=0.02496  (pojava=809)
  26  skor=0.6704402833  freq=0.02681  (pojava=869)
  09  skor=0.6650919779  freq=0.02601  (pojava=843)
  02  skor=0.6293915217  freq=0.02542  (pojava=824)
  21  skor=0.6231011851  freq=0.02549  (pojava=826)
  29  skor=0.6031615960  freq=0.02616  (pojava=848)


Tabela pravih frekvencija (opadajuce po freq, pa po broju):
  broj | pojava |   udeo
  -----+--------+--------
   08  |   910  | 0.02808
   23  |   905  | 0.02792
   34  |   873  | 0.02694
   26  |   869  | 0.02681
   37  |   860  | 0.02654
   11  |   860  | 0.02654
   32  |   857  | 0.02644
   33  |   854  | 0.02635
   22  |   851  | 0.02626
   39  |   849  | 0.02620
   29  |   848  | 0.02616
   10  |   845  | 0.02607
   35  |   843  | 0.02601
   09  |   843  | 0.02601
   38  |   842  | 0.02598
   07  |   842  | 0.02598
   24  |   840  | 0.02592
   25  |   839  | 0.02589
   16  |   837  | 0.02583
   31  |   830  | 0.02561
   13  |   828  | 0.02555
   05  |   828  | 0.02555
   21  |   826  | 0.02549
   03  |   825  | 0.02546
   02  |   824  | 0.02542
   28  |   820  | 0.02530
   18  |   820  | 0.02530
   06  |   816  | 0.02518
   19  |   813  | 0.02508
   04  |   812  | 0.02505
   12  |   810  | 0.02499
   14  |   809  | 0.02496
   15  |   797  | 0.02459
   27  |   788  | 0.02431
   01  |   788  | 0.02431
   30  |   787  | 0.02428
   36  |   786  | 0.02425
   20  |   770  | 0.02376
   17  |   766  | 0.02363
  ukupno pojava: 32410


Predlozene kombinacije (rangirane po skoru kombinacije):
  01. 03 06 22 26 29 36 37  skor_komb=4.3690554210
  02. 02 09 19 22 30 31 38  skor_komb=4.1325227180
  03. 07 09 14 15 21 24 31  skor_komb=3.9970385748
  04. 08 09 19 21 26 30 37  skor_komb=3.9622010365
  05. 05 06 13 21 22 23 31  skor_komb=3.8252379334
  06. 06 13 15 23 26 30 31  skor_komb=3.5941731725
  07. 01 03 08 12 13 19 38  skor_komb=3.5422842878
  08. 16 17 19 22 27 37 39  skor_komb=3.3724787688
  09. 09 13 21 28 29 33 38  skor_komb=3.3371805239
  10. 04 10 12 17 31 34 38  skor_komb=3.2695952049

Sacuvano: /Tesla/tesla_pycharge_2B.txt
"""



"""
pycharge motor je dao potpuno drugačiji profil od 1 i 2A.
Top lista je drugačija: 22, 31, 03, 37, 14, 26...
14 je opet jak iako ima malu frekvenciju (809) — dobar znak da talasni deo zaista menja rang.
Favorit 2B je: 03 06 22 26 29 36 37 sa skor_komb=4.3690554210.

Grupa 1: ručni SLW
2A: k-wave FDTD
2B: pycharge EM potencijal
"""



"""
Postoji teorijska osnova (EED/SLW) + alati za simulaciju talasa. 


GRUPA 2 

2. Gotove biblioteke za simulaciju talasa/EM polja (na njima bi se gradilo)


2A k-wave-python — simulacija talasnih polja (akustika, FDTD)

2B pycharge — EM polja/potencijali pokretnih naboja (JAX, GPU)

2C Wakis — 3D EM solver (računa i longitudinalne komponente)
           (najbliže Teslinom SLW)

2D rfx — diferencijabilni 3D FDTD EM simulator
         (može učenje/optimizacija)

k-wave-python — prvo. Najbliže grupi 1 (FDTD talasno polje).
pycharge — drugo. Uvodi prava polja naboja (JAX/GPU), dobra provera da li „izvor" menja rezultat.
Wakis — treće. Pravi 3D EM solver sa longitudinalnim komponentama → ovo je srce Tesline SLW priče.
rfx — poslednje. Diferencijabilni FDTD → kad sve radi, njime optimizujem parametre (učenje težina, ne ručno 0.7/0.3).

Logika: 
prve dve daju temelj i poređenje, treća donosi pravi longitudinalni talas, četvrta pretvara ceo sistem u nešto što se može podešavati/učiti.

Svaka varijanta = ista struktura kao grupa 1 (motor → primena na 4630 → skor → rangirane kombinacije), samo jači motor.
"""



"""
Analiza — Tesla 2B (pycharge EM potencijal)

Motor: pycharge, oscilujući tačkasti naboj. 
Ovde više nije samo talasna jednačina, nego EM izvor: skalarni potencijal S(x) duž x-pravca i iz njega E_x = -dS/dx. 
Ovo je konceptualno najbliže „izvor → polje → longitudinalni pravac" od ove prve tri verzije.

Mere polja: max S ≈ 0.99999, max |E_x| ≈ 17.60, ukupna energija ≈ 50591.86.

To je mnogo oštrije od Tesla 1 (E_x ≈ 0.10) — pycharge pravi jak gradijent, zato su rezultati drugačiji.

Top brojevi (talas + freq, 0.7/0.3): 22 (0.847) · 31 (0.833) · 03 (0.729) · 37 (0.707) · 14 (0.675) · 26 (0.670) · 09 (0.665) · 02 (0.629) · 21 (0.623) · 29 (0.603)

22, 31, 03 skaču visoko, iako nisu najjači frekvencijski brojevi → EM potencijal daje drugačiji talasni potpis.
14 je opet u top 10, kao i kod 2A → to je zanimljiv stabilan ne-frekvencijski kandidat.
34, koji je dominirao u Tesla 1 i 2A, kod 2B više nije top 10 → pycharge motor zaista daje zaseban signal.
37 i 26 su podržani i frekvencijom i talasom, pa su „kompromisno" jaki.
Favorit kombinacija: 03 06 22 26 29 36 37 (skor_komb = 4.3691). Ovo je trenutno najviši skor kombinacije od sve tri verzije, ali treba paziti: skale nisu direktno fizički uporedive među motorima, poredi se pre svega rang unutar modela.

Zaključak: 
2B je najagresivniji od prve tri verzije. 
Daje oštrije polje i drugačiju selekciju brojeva. 
Posebno su zanimljivi preseci: 
14 se pojavljuje i u 2A i u 2B, a 37, 31, 22 dobijaju jak signal iz EM pristupa.

stabilan kroz 1/2A: 34
stabilan kroz 2A/2B: 14
jak u 2B: 22, 31, 03, 37
"""



"""
source ~/tesla_env/bin/activate

Bitne verzije za tesla_env:

Paket	Verzija
python  3.11.13
numpy   2.2.6
scipy   1.15.3
pandas  3.0.3
matplotlib    3.10.9
k-Wave-python 0.6.2
pycharge      2.0.1
jax        0.10.1
jaxlib     0.10.1
jaxtyping  0.3.7
equinox    0.13.8
lineax     0.1.1
optimistix 0.1.0
ml-dtypes
(uz jax)
opencv-python 4.13.0.92
h5py          3.16.0
"""
