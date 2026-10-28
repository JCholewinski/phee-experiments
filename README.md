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

### 1. Maszyna

- GPU 24 GB (RTX 3090 / 4090 / 5090). Dla **RTX 5090 obraz musi mieć CUDA ≥ 12.8**
  i PyTorch ≥ 2.7, np. `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- Dysk: ok. 80 GB (modele z Hugging Face, cache, wyniki).
- Dla wygody ustaw cache Hugging Face na dysku z miejscem: `export HF_HOME=/workspace/hf`.

### 2. Kod i środowisko treningowe

```bash
cd /workspace
git clone https://github.com/JCholewinski/phee-experiments.git
cd phee-experiments
git checkout experiments-v2
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Dane są już w repozytorium w postaci po konwersji. Gdybyś chciał je wygenerować od nowa:
`python scripts/prepare_data.py`.

**Dostęp do T5Gemma**: modele `google/t5gemma-*` wymagają akceptacji licencji Gemma na
stronie modelu w Hugging Face (raz, na koncie). Potem zaloguj się tokenem na maszynie:

```bash
hf auth login          # starsze wersje huggingface_hub: huggingface-cli login
```

Pozostałe modele (BioBERT, Bio-ModernBERT, SciFive) są dostępne bez logowania.

### 3. Test na małych modelach (opcjonalnie, kilka minut, CPU wystarczy)

Sprawdza, czy wszystkie warianty przechodzą cały pipeline (trening, early stopping,
predykcja, ewaluacja). Wyniki są bez znaczenia, bo modele są losowe.

```bash
pip install tokenizers sentencepiece
python scripts/smoke/make_tiny_models.py
python scripts/smoke/run_smoke.py
```

### 4. Szybki test na prawdziwych modelach (GPU)

Uruchom po jednym przebiegu każdej metody na 200 zdaniach i 1 epoce. Pozwala sprawdzić,
czy modele się pobierają i ile trwa epoka (żeby zdecydować o rozmiarze Base / Large).

```bash
for c in seq/authors_baseline seq/own_crf eqa/authors_baseline eqa/own_current_predicted \
         gqa/authors_baseline gqa/own_current_predicted; do
  python scripts/run_experiment.py --config configs/$c.yaml --seed 1 --output_dir results_quick \
    --set data.limit=200 --set training.max_epochs=1 --force
done
```

Czas epoki jest w logu (`... epoch 1: train loss ..., dev f1 ...`) i w
`results_quick/<eksperyment>/seed1/training.json` (`train_seconds`).

### 5. Pełne eksperymenty

```bash
# kolejność: baseline'y, SL, extractive QA, generative QA (3 seedy każdy)
nohup python scripts/run_matrix.py --groups baselines seq eqa gqa --seeds 1 2 3 > matrix.log 2>&1 &
tail -f matrix.log
python scripts/run_matrix.py --list          # co zrobione, co zostało
```

- Każdy przebieg działa w osobnym procesie; log: `results/logs/<eksperyment>__seed<N>.log`.
- Skończony przebieg ma plik `DONE` i jest pomijany, więc po przerwaniu (np. restart
  instancji) wystarczy uruchomić to samo polecenie ponownie.
- Nieudany przebieg ma plik `FAILED` z tracebackiem, a matrix przechodzi do następnego.

### 6. Część LLM (osobne środowisko)

vLLM instaluje własne wersje torch i transformers, dlatego potrzebny jest osobny venv:

```bash
python -m venv /workspace/venv-llm
/workspace/venv-llm/bin/pip install -r requirements-llm.txt
# Llama 3.1 jest modelem z ograniczonym dostępem: zaakceptuj licencję na HF, potem:
/workspace/venv-llm/bin/huggingface-cli login
PYTHONPATH=$PWD /workspace/venv-llm/bin/python scripts/run_matrix.py --groups llm
```

Domyślne modele: `Qwen/Qwen3-8B` (bez trybu „thinking”) i
`meta-llama/Llama-3.1-8B-Instruct`. Oba mieszczą się w 24 GB w bf16. Inne modele
dodajesz kopiując `configs/llm/qwen3_8b.yaml` i dopisując go w `configs/matrix.yaml`
(większe modele: `model.quantization: fp8` lub gotowe wersje AWQ).

### 7. Tabele

```bash
python scripts/aggregate_results.py      # results/tables/*.md, *.csv, *.png
python scripts/tag_ceiling.py            # górne granice schematów SL
```

## Dane

- **Poprawiona wersja PHEE** z repozytorium Sun i in. (2024): te same 4827 zdań i
  oficjalny podział po dokumentach (2898 / 961 / 968), poprawione adnotacje
  `Subject.Disorder`, `Time_elapsed`, `Duration`.
- Tokenizacja (spaCy) pochodzi z oryginalnego wydania PHEE, więc SL działa na tych samych
  tokenach co u autorów. Wszystkie zdania zostały dopasowane bez błędów; 135 z ok.
  34 tys. granic adnotacji wypada w środku tokenu (np. „mofeti” zamiast „mofetil”).
- `Combination.Drug` oraz atrybuty zdarzeń (negacja, spekulacja, nasilenie) są pominięte.
- Adnotacja nieciągła (kilka części) to kilka spanów, jak w ewaluacji autorów.

Statystyki konwersji: `data/phee2/processed/conversion_stats.json`.

## Ewaluacja

`phee/evaluation.py`, ta sama dla wszystkich metod:

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

### Sequence labeling (`configs/seq/`)

| Config | Opis |
|---|---|
| `authors_baseline` | jak u autorów: jeden model, etykiety złożone (argument + subargument), schemat IO, zdanie kopiowane osobno dla każdego zdarzenia, BioBERT, głowica liniowa |
| `own_linear` / `own_mlp` / `own_crf` | Twoja konstrukcja: dwa modele (poziom główny i subargumenty), BIO, zdanie z kilkoma zdarzeniami użyte raz, Bio-ModernBERT; głowice: liniowa, MLP (256, GELU), CRF z **uczonymi** przejściami (NLL CRF, ograniczenia BIO) |

Dekodowanie (`argmax`, `viterbi` z ograniczeniami BIO) jest liczone na tym samym
wytrenowanym modelu, więc każdy przebieg daje oba warianty. Spany nakładające się na span
o wyższym priorytecie są pomijane w całości (trigger > TREATMENT > EFFECT > SUBJECT,
dla subargumentów częstsza etykieta wygrywa).

### Extractive QA (`configs/eqa/`)

Pipeline: klasyfikator typu zdarzenia → pytanie o trigger dla każdego typu → pytania o
SUBJECT / TREATMENT / EFFECT dla każdego triggera → pytania o subargumenty dla każdego
argumentu głównego. Każdy etap ma własny model i własny early stopping.

| Config | Szablony | Trening etapów 3–4 |
|---|---|---|
| `authors_baseline` | autorów (EEQA) | na danych referencyjnych, BioBERT |
| `own_current_predicted` (bazowy) | obecne | na predykcjach poprzedniego etapu |
| `own_current_gold` | obecne | na danych referencyjnych |
| `own_authors_predicted` | autorów | na predykcjach |
| `own_definitions_predicted` | obecne + definicja etykiety | na predykcjach |

Pytanie może mieć kilka odpowiedzi (powyżej wyniku „brak odpowiedzi”, maks. 3).

### Generative QA (`configs/gqa/`)

Etap 1 generuje zdarzenia (trigger + argumenty główne), etap 2 odpowiada na pytania o
subargumenty. Dekodowanie `free` i `constrained` jest liczone na tych samych modelach.

| Config | Format celu | Szablony etapu 2 | Trening etapu 2 | Model |
|---|---|---|---|---|
| `authors_baseline` | autorów | autorów | referencyjne | SciFive-base-PMC |
| `own_current_predicted` (bazowy) | obecny | obecne | predykcje etapu 1 | T5Gemma b-b UL2 |
| `own_current_gold` | obecny | obecne | referencyjne | T5Gemma |
| `own_authors_predicted` | obecny | autorów | predykcje | T5Gemma |
| `own_definitions_predicted` | obecny | obecne + definicje | predykcje | T5Gemma |

Dekodowanie ograniczone: każdy generowany fragment musi być fragmentem zdania (między
granicami słów), a struktura wyjścia musi pasować do gramatyki znaczników (`phee/gqa/constrained.py`).

### LLM (`configs/llm/`)

Prompt ze schematem i definicjami etykiet, odpowiedź w JSON; wariant 0-shot i 5-shot
(przykłady z train dobierane przez BM25, jak u Sun i in. 2024). Dekodowanie zachłanne,
jeden przebieg. Raportowany jest też odsetek odpowiedzi w złym formacie.

## Wyniki i analizy

```
results/<eksperyment>/seed<N>/
  summary.json            podsumowanie wszystkich wariantów, koszt, środowisko
  training.json           każdy etap: najlepsza epoka, czas, pamięć GPU, liczba parametrów
  history/<etap>.json     przebieg treningu (loss, F1 na dev po każdej epoce)
  metrics/<wariant>__<split>.json      pełne metryki
  predictions/<wariant>__<split>.jsonl predykcje (z surowym tekstem dla GQA/LLM)
results/tables/           wyniki zagregowane (średnia ± odchylenie po seedach)
```

Koszt: czas treningu (bez ewaluacji na dev, osobno czas ewaluacji), czas inferencji na
zdanie na zbiorze testowym (wszystkie etapy pipeline'u), szczytowa pamięć GPU.

## Różnice względem artykułu PHEE i ograniczenia

- **Metryka**: bez typu zdarzenia przy argumentach i na unikalnych parach (autorzy: z typem,
  każde wystąpienie osobno); dla SL autorów usuwa to karę za ostatni trigger w zdaniu.
- **Dane**: poprawiona wersja z 2024 r.; bez `Combination.Drug`.
- **Baseline SL**: bez frameworku ACE (autorzy łączyli embeddingi kilku modeli),
  BioBERT z głowicą liniową.
- **Baseline extractive QA**: triggery w naszej formule QA (autorzy: klasyfikacja tokenów
  EEQA), pozostałe etapy z szablonami i ustawieniami autorów.
- **Ograniczenie SL**: w poprawionych danych 1185 spanów ma jednocześnie etykietę
  `SUBJECT_DISORDER` i `TREATMENT_DISORDER`, a 206 `DURATION` i `TIME_ELAPSED`. Jedna
  etykieta na token pozwala oddać tylko jedną z nich; górna granica SL dla subargumentów
  to ok. 93.7 F1 (`scripts/tag_ceiling.py`).
- **Predykcje na zbiorze treningowym** (tryb „predicted” w QA) pochodzą z modelu
  trenowanego na tych samych danych, więc są lepsze niż na teście.
- SL nie przypisuje argumentów do zdarzeń; metryka tego nie ocenia (jak u autorów).
