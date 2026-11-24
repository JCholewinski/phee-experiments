# PHEE: porównanie metod ekstrakcji informacji

Branch `experiments-v2` realizuje plan prac z dokumentu „Plan prac: ulepszenia metod
ekstrakcji informacji (PHEE)”. Wszystkie metody (sequence labeling, extractive QA,
generative QA, LLM zero/few-shot) korzystają z tych samych danych, tej samej procedury
treningu i tej samej ewaluacji. Poprzedni kod jest w `legacy/` (bez zmian, do wglądu).

## Spis treści

1. [Struktura repozytorium](#struktura-repozytorium)
2. [Uruchomienie na vast.ai krok po kroku](#uruchomienie-na-vastai-krok-po-kroku)
3. [Dane](#dane)
4. [Ewaluacja](#ewaluacja)
5. [Metody i warianty](#metody-i-warianty)
6. [Wyniki i analizy](#wyniki-i-analizy)
7. [Różnice względem artykułu PHEE i ograniczenia](#różnice-względem-artykułu-phee-i-ograniczenia)

## Struktura repozytorium

```
phee/                    kod (pakiet Pythona)
  data.py                konwersja poprawionej wersji PHEE do wspólnego formatu
  evaluation.py          ewaluacja tekstowa (wspólna dla wszystkich metod)
  training.py            wspólna procedura treningu (early stopping, seedy, bf16)
  seq/                   sequence labeling (schematy etykiet, głowice, dekodowanie)
  eqa/                   extractive QA (klasyfikator typu → trigger → argumenty → subargumenty)
  gqa/                   generative QA (etap 1, etap 2, dekodowanie ograniczone)
  llm/                   LLM zero/few-shot (prompty, BM25, vLLM)
  analysis.py            agregacja wyników po seedach
configs/                 konfiguracje eksperymentów (YAML, dziedziczenie przez `inherit:`)
  matrix.yaml            grupy eksperymentów do uruchomienia
scripts/
  prepare_data.py        konwersja danych (wynik jest już w repo)
  run_experiment.py      jeden eksperyment, jeden seed
  run_matrix.py          wiele eksperymentów × seedy, wznawianie po przerwaniu
  aggregate_results.py   tabele i wykres
  tag_ceiling.py         górne granice schematów etykiet SL
  smoke/                 test całości na małych losowych modelach (CPU, offline)
data/
  phee2/source/          poprawione adnotacje (Sun i in. 2024, licencja MIT)
  phee2/processed/       dane po konwersji (train/dev/test.jsonl)
  phee_v1_tokenized/     oryginalne PHEE w formacie EEQA (z niego bierzemy tokenizację)
legacy/                  poprzednia implementacja
```

## Uruchomienie na vast.ai krok po kroku

Plan: dla każdej metody dwie wersje na **tym samym modelu** (baseline autorów i wersja
rozwinięta) oraz jeden dodatkowy test LLM (przykłady tylko z 50 oznaczonych zdań).
Najpierw **tryb testowy** (seed 1 każdego przebiegu), sprawdzenie wyników, potem
**wyniki końcowe** (seedy 2 i 3; przebiegi z seedem 1 są wykorzystane ponownie).

### 1. Maszyna

- GPU 24 GB (RTX 3090 / 4090 / 5090). Dla **RTX 5090 obraz musi mieć CUDA ≥ 12.8**
  i PyTorch ≥ 2.7, np. `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- Dysk: ok. 80 GB (modele z Hugging Face, cache, wyniki).
- **Cały plan na tej samej karcie** (tryb testowy i końcowy), inaczej czasy i koszty nie
  są porównywalne; tabele ostrzegają, jeśli przebiegi pochodzą z różnych GPU.
- Zanotuj cenę instancji (USD/h, widoczna na vast.ai): podajesz ją jako `GPU_PRICE`.

### 2. Kod i środowiska

```bash
cd /workspace
git clone https://github.com/JCholewinski/phee-experiments.git
cd phee-experiments
git checkout experiments-v2
bash scripts/vast/setup.sh --llm   # zależności + testy; --llm: osobny venv z vLLM (/workspace/venv-llm)
source ~/.bashrc
export HF_HOME=/workspace/hf PYTHONPATH=$PWD
```

vLLM instaluje własne wersje torch i transformers, dlatego część LLM ma osobny venv;
skrypty same wybierają właściwe środowisko. Wszystkie modele planu (BioBERT, SciFive,
Qwen3-8B) są dostępne bez logowania do Hugging Face. Dane są już w repozytorium po
konwersji (od nowa: `python scripts/prepare_data.py`).

### 3. Testy (opcjonalnie, CPU wystarczy)

Testy logiki niezależnej od modeli (ewaluacja, schematy etykiet, parsery, CRF i Viterbi
porównane z pełnym przeglądem ścieżek, ograniczone dekodowanie na tokenizerach):
`python -m pytest tests -q` (uruchamia je też `setup.sh`).

Test całego pipeline'u na małych, losowych modelach (kilka minut, wyniki bez znaczenia):

```bash
python scripts/smoke/make_tiny_models.py
python scripts/smoke/run_smoke.py
```

### 4. Szybki test na prawdziwych modelach (ok. 20–30 min)

```bash
nohup bash scripts/vast/quick_check.sh > quick.log 2>&1 &
tail -f quick.log
```

Po jednym krótkim przebiegu każdej metody planu (200 zdań, 1 epoka; LLM: 20 zdań, 0- i
5-shot). Sprawdza, czy modele się pobierają i działają na GPU; na końcu wypisuje czas
epoki przeliczony na pełne dane, pamięć GPU i czas LLM na zdanie. Wyniki trafiają do
`results_quick/` (ich jakość jest bez znaczenia).

### 5. Tryb testowy (seed 1)

```bash
GPU_PRICE=0.25 nohup bash scripts/vast/run_plan.sh test > plan_test.log 2>&1 &
tail -f plan_test.log
```

`GPU_PRICE` to cena Twojej instancji w USD/h. Kolejność: część LLM (baseline z promptem
Sun i in., wersja rozwinięta, test z 50 zdaniami), potem metody trenowane (grupy
`baselines` i `improved`), na końcu tabele. Orientacyjnie na RTX 3090 ok. 6 h.

| Grupa (`configs/matrix.yaml`) | Przebiegi |
|---|---|
| `llm` | `qwen3_8b_authors_prompt` (baseline), `qwen3_8b` (rozwinięta): 0- i 5-shot w jednym przebiegu, deterministyczne, tylko seed 1 |
| `llm_pool50` | `qwen3_8b_pool50`: 5-shot, przykłady z 50 losowych zdań (inne 50 dla każdego seeda) |
| `baselines` | `seq/authors_baseline`, `eqa/authors_baseline`, `gqa/authors_baseline` |
| `improved` | `seq/own_crf`, `eqa/own_current_predicted`, `gqa/own_current_predicted` |

### 6. Sprawdzenie wyników trybu testowego

W `results/tables/`:

- `comparison_test.md`: dla każdej metody baseline → wersja rozwinięta (argumenty, rdzeń
  zdarzenia, argumenty w zdarzeniu, to samo na zdaniach z kilkoma zdarzeniami, koszt) oraz
  test z 50 zdaniami;
- `main_test.md`: metryki główne wszystkich wariantów i koszt przebiegu w USD;
- `details_test.md`: triggery, subargumenty, token-F1, czasy, pamięć GPU;
- `subsets_test.md`, `labels_test.md`: zdania z jednym / kilkoma zdarzeniami, każda etykieta;
- `checks.md`: na co zwrócić uwagę (nieudane przebiegi, etap najlepszy w ostatniej
  epoce, bardzo niskie wyniki, odpowiedzi LLM w złym formacie, ucięte generowanie).

Punkty odniesienia: baseline extractive QA i generative QA nie zmieniły się od 1. rundy
(test, seed 1: argumenty główne 55,0 i 72,1 EM-F1), więc powinny wyjść bardzo podobnie.
Baseline SL ma teraz warstwę CRF (1. runda bez CRF: 63,2).

Zmiana konfiguracji po trybie testowym jest wykrywana automatycznie: przebieg, którego
zapisany `config.json` różni się od obecnej konfiguracji, liczy się od nowa, a stary
katalog trafia do `results/_replaced/` (dotyczy też wyników 1. rundy, jeśli są jeszcze w
`results/` na maszynie). Jeśli zmieni się tylko kod metody, usuń jej katalogi z
`results/` (np. `rm -r results/gqa__own_current_predicted`).

### 7. Wyniki końcowe (seedy 2 i 3)

```bash
GPU_PRICE=0.25 nohup bash scripts/vast/run_plan.sh final > plan_final.log 2>&1 &
```

Uruchamia seedy 2 i 3 metod trenowanych i testu z 50 zdaniami (LLM z całą pulą jest
deterministyczny i liczony raz), potem tabele ze średnią ± odchyleniem z 3 seedów.
Orientacyjnie ok. 9–10 h na RTX 3090.

- Skończony przebieg ma plik `DONE` i jest pomijany, więc po przerwaniu (np. restart
  instancji) wystarczy uruchomić to samo polecenie ponownie.
- Nieudany przebieg ma plik `FAILED` z tracebackiem, a plan przechodzi do następnego.
- Log każdego przebiegu: `results/logs/<eksperyment>__seed<N>.log`; stan planu:
  `python scripts/run_matrix.py --groups llm llm_pool50 baselines improved --seeds 1 2 3 --list`.
- Tabele można przeliczyć w każdej chwili: `python scripts/aggregate_results.py --gpu_price 0.25`.
- Pobranie wyników na swój komputer: `tar czf results.tgz results` i np. `scp` albo
  pobranie pliku w Jupyterze vast.ai.

Pozostałe zaimplementowane warianty (inne głowice SL, szablony i trening na danych
referencyjnych w QA, Llama 3.1) są w grupach `extra_*` w `configs/matrix.yaml` i nie są
częścią planu, np. `python scripts/run_matrix.py --groups extra_seq_heads --seeds 1`.

## Dane

- **Poprawiona wersja PHEE** z repozytorium Sun i in. (2024): te same 4827 zdań i
  oficjalny podział po dokumentach (2898 / 961 / 968), poprawione adnotacje
  `Subject.Disorder`, `Time_elapsed`, `Duration`.
- Tokenizacja (spaCy) pochodzi z oryginalnego wydania PHEE, więc SL działa na tych samych
  tokenach co u autorów. Wszystkie zdania zostały dopasowane bez błędów; 135 z ok.
  34 tys. granic adnotacji wypada w środku tokenu (np. „mofeti” zamiast „mofetil”).
  Usunięte są tokeny złożone z samych spacji lub znaku nowej linii (6 w całym zbiorze),
  które rozbijałyby span w SL na dwie części; adnotacje się nie zmieniają.
- `Combination.Drug` oraz atrybuty zdarzeń (negacja, spekulacja, nasilenie) są pominięte.
- Adnotacja nieciągła (kilka części) to kilka spanów, jak w ewaluacji autorów.

Statystyki konwersji: `data/phee2/processed/conversion_stats.json`.

## Ewaluacja

`phee/evaluation.py` i `phee/events.py`, ta sama dla wszystkich metod.

**Metryki główne** (tabela `main_*.md`, z nich wyciągamy wnioski):

| Metryka | Co mierzy |
|---|---|
| **Argumenty F1** | argumenty główne jako pary (etykieta, tekst) w zdaniu, jak w artykule PHEE; nie sprawdza, do którego zdarzenia trafił argument |
| **Rdzeń zdarzenia F1** | pary (typ zdarzenia, lek) dla każdego zdarzenia; lek = Treatment.Drug (bez leku: tekst TREATMENT); kilka leków = kilka par; zdarzenia o tym samym rdzeniu łączone |
| **Argumenty w zdarzeniu F1** | (typ, lek, etykieta, tekst): argument (w tym efekt, pacjent, subargumenty) liczy się tylko w zdarzeniu z poprawnym rdzeniem; ogółem i osobno na zdaniach z kilkoma zdarzeniami |
| **Koszt** | czas całego przebiegu i jego koszt w USD |

Wszystkie z dokładnym dopasowaniem znormalizowanego tekstu (EM). Każda metoda zapisuje w
predykcjach numer zdarzenia każdego elementu (`"event"`) i jego typ (`"event_type"`), więc
metryki zdarzeń i późniejsza analiza błędów liczą się z zapisanych plików. Sequence
labeling w wersji bazowej oznacza tylko fragmenty; zdarzenia buduje reguła: jedno
zdarzenie na trigger, argument do najbliższego triggera, subargument do zawierającego go
argumentu głównego (`assign_by_nearest_trigger`). Wersja rozwinięta taguje argumenty osobno
dla każdego triggera zaznaczonego znacznikami `[TRG] ... [/TRG]`, więc przypisanie do
zdarzenia daje sam model. Górna granica przy idealnych etykietach (zdania z kilkoma
zdarzeniami, argumenty w zdarzeniu): reguła 71,2 (format dwupoziomowy) i 67,3 (format
autorów), tagowanie z `[TRG]` 92,9.

**Diagnostyka** (`details_*.md`, `subsets_*.md`, `labels_*.md`):

- predykcja to para **(etykieta, znormalizowany tekst)**; normalizacja jak w oficjalnym
  ewaluatorze autorów (małe litery, interpunkcja jako granica, bez a/an/the);
- liczymy **unikalne pary w zdaniu**: argument wspólny dla dwóch zdarzeń liczy się raz,
  przypisanie argumentu do konkretnego zdarzenia nie jest oceniane;
- argumenty **bez wymogu zgodności typu zdarzenia**; triggery z typem (klasyfikacja) i bez
  (identyfikacja);
- **EM-F1** (dokładne dopasowanie tekstu) i **token-F1** (jak u autorów);
- wyniki: triggery, typ zdarzenia, argumenty główne, subargumenty, wszystko razem, każda
  etykieta osobno, osobno zdania z jednym i z kilkoma zdarzeniami;
- **zależność od triggera**: dla każdego argumentu tabela 2×2 (trigger zdarzenia znaleziony
  / nie) × (argument znaleziony / nie), analogicznie subargument × argument nadrzędny.
- generative QA i LLM są oceniane na **surowym wygenerowanym tekście** (bez dopasowania do
  zdania).

Sprawdzone na zdaniach z jednym zdarzeniem: wynik jest identyczny z oficjalnym
ewaluatorem autorów (różnice tylko tam, gdzie autorzy liczą powtórzenia).

## Metody i warianty

Wspólna procedura treningu (`phee/training.py`): AdamW, rozgrzewka 10%, liniowy spadek
LR, bf16, do 20 epok, **early stopping po 2 ocenach bez poprawy** wyniku F1 na dev
(każdy etap ma własną miarę na dev), przywrócenie najlepszego checkpointu, 3 seedy.

Modele: **ten sam model w obu wersjach metody**, żeby różnica wynikała ze zmian w
konstrukcji metody, a nie z lepszego modelu; ustawienia treningu (LR, batch) też są takie
same jak w baseline.

| Metoda | Model (baseline i wersja rozwinięta) |
|---|---|
| Sequence labeling | `dmis-lab/biobert-base-cased-v1.1` |
| Extractive QA | `dmis-lab/biobert-base-cased-v1.1` |
| Generative QA | `razent/SciFive-base-PMC` |
| LLM | `Qwen/Qwen3-8B` (opcjonalnie `meta-llama/Llama-3.1-8B-Instruct`, grupa `extra_llm`) |

W 1. rundzie wersje rozwinięte używały BioClinical ModernBERT i T5Gemma; to mieszało
efekt zmian z efektem modelu, dlatego od 2. rundy model jest wspólny.

### Sequence labeling (`configs/seq/`)

| Config | Opis |
|---|---|
| `authors_baseline` | jak u autorów: jeden model, etykiety złożone (argument + subargument), schemat IO, zdanie kopiowane osobno dla każdego zdarzenia, warstwa CRF z uczonymi przejściami (dekodowanie Viterbi), BioBERT |
| `own_crf` (wersja rozwinięta) | trzy modele: tagger triggerów (z typem ADE/PTE), potem tagger argumentów głównych i tagger subargumentów uruchamiane **osobno dla każdego triggera**, zaznaczonego w zdaniu znacznikami `[TRG] ... [/TRG]` bez typu (jak MAILEX, Srivastava i in. 2023); trening: jeden przykład na trigger ze zbioru treningowego; BIO, CRF z uczonymi przejściami i **zakazem przejść niemożliwych w BIO** w treningu i dekodowaniu (Viterbi), BioBERT |
| `own_crf_nearest` (extra, ablacja) | ten sam tagger CRF, ale jedno otagowanie całego zdania (dwa modele: poziom główny i subargumenty) i zdarzenia z reguły najbliższego triggera, jak w baseline; pokazuje, ile daje tagowanie z `[TRG]` |
| `own_linear` / `own_mlp` (extra) | ta sama konstrukcja z głowicą liniową / MLP (256, GELU), dekodowanie argmax i Viterbi z ograniczeniami BIO |

Dekodowanie (`argmax`, `viterbi`) jest liczone na tym samym wytrenowanym modelu, więc
przebieg z kilkoma dekodowaniami daje kilka wariantów. Spany nakładające się na span
o wyższym priorytecie są pomijane w całości (trigger > TREATMENT > EFFECT > SUBJECT,
dla subargumentów częstsza etykieta wygrywa).

### Extractive QA (`configs/eqa/`)

Własny pipeline: klasyfikator typu zdarzenia → pytanie o trigger dla każdego typu →
pytania o SUBJECT / TREATMENT / EFFECT dla każdego triggera → pytania o subargumenty dla
każdego argumentu głównego. Każdy etap ma własny model i własny early stopping.

Baseline autorów szuka triggerów jak w artykule (EEQA): tagger słów, który pierwszemu
słowu każdego triggera nadaje typ zdarzenia (O / ADE / PTE); każde oznaczone słowo to
trigger. Klasyfikatora typu zdarzenia wtedy nie ma.

| Config | Triggery | Szablony | Trening etapów 3–4 |
|---|---|---|---|
| `authors_baseline` | tagger słów (EEQA) | autorów | na danych referencyjnych |
| `own_current_predicted` (bazowy) | pytanie | obecne | na predykcjach poprzedniego etapu |
| `own_current_gold` | pytanie | obecne | na danych referencyjnych |
| `own_authors_predicted` | pytanie | autorów | na predykcjach |
| `own_definitions_predicted` | pytanie | obecne + definicja etykiety | na predykcjach |

Wybór odpowiedzi w obu wersjach jak w kodzie autorów (`selection: authors`, maks. 30
tokenów): początek i koniec osobno powyżej wyniku „brak odpowiedzi” ([CLS]), tylko 2
najlepszych kandydatów na pytanie, kandydat odrzucany, gdy nachodzi na fragment już
przyjęty w zdaniu dla tego samego typu zdarzenia i roli. Wersja rozwinięta różni się od
baseline'u tylko trzema zmianami: triggery (klasyfikator typu + pytanie o trigger),
własne szablony pytań, trening etapów argumentów na predykcjach. Pytanie o trigger
używa tej samej reguły, ale zawsze zwraca co najmniej jedną odpowiedź (klasyfikator już
stwierdził, że zdarzenie tego typu jest w zdaniu), więc w zdaniu można znaleźć dwa
zdarzenia tego samego typu. Żeby model nauczył się „braku
odpowiedzi” przy triggerach, w treningu są też pytania o typ zdarzenia, którego w zdaniu
nie ma.

### Generative QA (`configs/gqa/`)

Etap 1 generuje zdarzenia (trigger + argumenty główne), etap 2 odpowiada na pytania o
subargumenty. Dekodowanie `free` i `constrained` jest liczone na tych samych modelach;
wersja rozwinięta to wariant `constrained`. Wszystkie warianty: SciFive-base-PMC, LR 5e-4
(etap 1) / 5e-5 (etap 2), efektywny batch 16, beam 3.

| Config | Format celu | Szablony etapu 2 | Trening etapu 2 |
|---|---|---|---|
| `authors_baseline` | autorów | autorów | referencyjne |
| `own_current_predicted` (wersja rozwinięta) | własny | obecne (wskazują argument) | predykcje etapu 1 |
| `own_current_gold` | własny | obecne | referencyjne |
| `own_authors_predicted` | własny | autorów | predykcje |
| `own_definitions_predicted` | własny | obecne + definicje | predykcje |

Dekodowanie ograniczone: każdy generowany fragment musi być fragmentem zdania (między
granicami słów), a struktura wyjścia musi pasować do gramatyki znaczników (`phee/gqa/constrained.py`).
Wyjście ucięte limitem długości nie dokleja niedokończonego znacznika (np. „[Eff”) do
odpowiedzi; liczba takich wyjść jest zapisywana (`stage1_truncated_outputs`,
`stage2_truncated_outputs` w `summary.json`).

### LLM (`configs/llm/`)

| Config | Opis |
|---|---|
| `qwen3_8b_authors_prompt` (baseline) | prompt Sun i in. (2024) z ich kodu: instrukcja „Explanation” (krótki opis każdego pola), ich format odpowiedzi i parsowanie; bez triggerów (oni o nie nie pytali), więc trigger F1 nie jest raportowany |
| `qwen3_8b` (wersja rozwinięta) | własny prompt: definicje argumentów i triggera, reguły oznaczania, odpowiedź JSON z triggerem |
| `qwen3_8b_pool50` (test dodatkowy) | jak `qwen3_8b`, 5-shot, ale przykłady wybierane tylko z 50 losowych zdań treningowych (inne 50 dla każdego seeda) |

Przykłady są dobierane **tak samo w obu wersjach**, jak u Sun i in. (2024): BM25,
5 najbardziej podobnych zdań z ADE i 5 z PTE (`demonstrations: per_type` w
`configs/llm/base.yaml`; `top` = 5 najbardziej podobnych zdań dowolnego typu). Dzięki temu
wersje różnią się tylko poleceniem. Warianty 0-shot i 5-shot (5 na typ) w jednym
przebiegu, dekodowanie zachłanne, więc jeden przebieg wystarcza. Raportowany jest też
odsetek odpowiedzi w złym formacie.

## Wyniki i analizy

```
results/<eksperyment>/seed<N>/
  summary.json            podsumowanie wszystkich wariantów, koszt, środowisko
  training.json           każdy etap: najlepsza epoka, czas, pamięć GPU, liczba parametrów
  history/<etap>.json     przebieg treningu (loss, F1 na dev po każdej epoce)
  metrics/<wariant>__<split>.json      pełne metryki
  predictions/<wariant>__<split>.jsonl predykcje z numerem zdarzenia każdego elementu
                                       (z surowym tekstem dla GQA/LLM)
results/tables/           wyniki zagregowane (średnia ± odchylenie po seedach)
```

Koszt: czas treningu (bez ewaluacji na dev, osobno czas ewaluacji), czas całego przebiegu
(`run_seconds`: wczytanie modelu, trening, ewaluacje, predykcje wszystkich wariantów) i
jego koszt w USD (czas × `--gpu_price`), czas inferencji na zdanie na zbiorze testowym
(wszystkie etapy pipeline'u), szczytowa pamięć GPU.

## Różnice względem artykułu PHEE i ograniczenia

- **Metryka**: bez typu zdarzenia przy argumentach i na unikalnych parach (autorzy: z typem,
  każde wystąpienie osobno); dla SL autorów usuwa to karę za ostatni trigger w zdaniu.
- **Dane**: poprawiona wersja z 2024 r.; bez `Combination.Drug`.
- **Baseline SL**: bez frameworku ACE (autorzy łączyli wektory BERT i BioBERT), jeden
  BioBERT z warstwą CRF, jak w końcowym systemie autorów.
- **Baseline extractive QA**: tagger triggerów jak w EEQA, ale bez stałego zapytania
  „null” na wejściu i z pełną tokenizacją słów (autorzy podawali modelowi tylko pierwszy
  sub-token każdego słowa); bez znacznika pozycji triggera w pytaniach o argumenty.
  Wybór odpowiedzi 1:1 z kodu autorów (`selection: authors`, `eeqa/code/run_stage1.py`,
  `run_stage2.py`): początek i koniec osobno powyżej wyniku [CLS], tylko 2 najlepszych
  kandydatów na pytanie, kandydat odrzucany, gdy jego początek lub koniec wpada w fragment
  już przyjęty w zdaniu dla tego samego typu zdarzenia i roli (także z pytań o inne
  triggery). Według ich `EXP_RECORD.md` ta reguła daje 65,7 z artykułu; pierwotna reguła
  EEQA dawała 57,4 przy precyzji 46. Do 22.11 baseline miał regułę „do 2 nienakładających
  się odpowiedzi powyżej [CLS]” i precyzję argumentów głównych 45 (test, seed 1).
- **Ograniczenie SL**: w poprawionych danych 1183 spany (239 w teście) mają jednocześnie etykietę
  `SUBJECT_DISORDER` i `TREATMENT_DISORDER`, a 206 `DURATION` i `TIME_ELAPSED`. Jedna
  etykieta na token pozwala oddać tylko jedną z nich; górna granica SL dla subargumentów
  to ok. 93.7 F1 (`scripts/tag_ceiling.py`).
- **Predykcje na zbiorze treningowym** (tryb „predicted” w QA) pochodzą z modelu
  trenowanego na tych samych danych, więc są lepsze niż na teście.
- Baseline SL nie przypisuje argumentów do zdarzeń; zdarzenia buduje prosta reguła
  (najbliższy trigger), więc jego metryki zdarzeń zależą też od tej reguły. Wersja
  rozwinięta (`[TRG]`) zależy od taggera triggerów: zgubiony trigger to zgubione zdarzenie.
