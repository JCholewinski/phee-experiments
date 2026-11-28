# Otwarte kwestie (do sprawdzenia później)

## Decyzje z 14.11.2026 (plan 2. rundy)

- Ten sam model w obu wersjach każdej metody: SL i extractive QA BioBERT v1.1, generative
  QA SciFive-base-PMC (LR jak w baseline), LLM Qwen3-8B. Ustawienia treningu jak w baseline.
- Baseline SL z warstwą CRF (jak końcowy system autorów), bez łączenia modeli w ACE.
- Wersja rozwinięta SL: CRF z zakazem przejść BIO, dekodowanie Viterbi (bez wariantu argmax).
- LLM: przykłady w obu wersjach jak u Sun i in. (BM25, 5 z ADE + 5 z PTE); wersje różnią
  się tylko promptem. Test dodatkowy: przykłady z 50 losowych zdań (inne dla każdego seeda).
  Bez porównania z SL trenowanym na 50 zdaniach.
- Uruchomienie: `scripts/vast/run_plan.sh test`, sprawdzenie, `run_plan.sh final`.
- Do decyzji: znacznik pozycji triggera w zdaniu w extractive QA (to samo słowo-trigger w
  dwóch miejscach albo jeden trigger dwóch zdarzeń: 5 z 40 zdań wielozdarzeniowych w teście).

## Decyzje z 19.11.2026 (metryki zdarzeń)

- Metryki główne: argumenty F1 (jak autorzy), rdzeń zdarzenia (typ, lek), argumenty w
  zdarzeniu (ogółem i na zdaniach z kilkoma zdarzeniami), koszt. Trigger nie jest w
  metrykach głównych. Token-F1, triggery, subargumenty, etykiety: diagnostyka.
- Predykcje zapisują numer zdarzenia każdego elementu (analiza błędów później, z plików).
- SL (decyzja 21.11): wersja rozwinięta taguje argumenty osobno dla każdego triggera,
  zaznaczonego w zdaniu znacznikami [TRG] ... [/TRG] bez typu zdarzenia (jak MAILEX,
  Srivastava i in. 2023). Baseline zostaje z regułą „najbliższy trigger” (autorzy w ogóle
  nie łączą argumentów w zdarzenia). Górna granica przy idealnych etykietach, zdania z
  kilkoma zdarzeniami, argumenty w zdarzeniu: reguła 71,2 -> [TRG] 92,9 (rdzeń 88,9 -> 100).
  Ablacja: `seq/own_crf_nearest` (ten sam tagger, reguła zamiast [TRG]) w extra_seq_heads.
- LLM (decyzja 24.11): nasz prompt -- subject jako JEDEN tekst (null, gdy brak), nowe
  definicje subject (jedna, najpełniejsza wzmianka; opisy tylko z tego fragmentu) i effect
  (cała fraza z określeniami; efekty wymienione razem = jeden fragment), tylko w prompcie
  LLM (`LLM_DEFINITIONS`). Powód: test 22.11, 5-shot, dev: precyzja SUBJECT 67 vs 73 u
  autorów (555 wypisanych przy 515 prawdziwych, słowa P 80 / R 83), EFFECT 65 vs 69 (te
  same słowa, inne granice). Adnotacje: wiek/płeć/rasa/liczba wewnątrz subject; efekty
  wymienione razem zwykle jednym fragmentem (17% zdarzeń z „and/or/,” w fragmencie vs 6% z
  kilkoma fragmentami). W przykładach przy kilku fragmentach subject: pierwszy (pacjent).
  Treatment bez zmian (precyzja jak u autorów).
- Trening na predykcjach (decyzja 28.11): czy wersje rozwinięte QA nie uczą się na złych
  ekstrakcjach? Grupa `train_inputs` w planie (3 seedy): `eqa/own_current_gold` i
  `gqa/own_current_gold` -- te same wersje rozwinięte, tylko późniejsze etapy trenowane na
  prawdziwych wejściach; w tabeli porównań osobne wiersze. Log 24.11 (generative QA, etap 2):
  29531 pytań z prawdziwych argumentów, z predykcji etapu 1 22262 identycznych (pominięte),
  4837 niedokładnych, 625 bez dopasowania. SL: trening na przewidzianych triggerach już
  porównany (22.11 złote -> 24.11 przewidziane: argumenty test 68,3 -> 72,6).
- LLM, wersja dwuetapowa (sprawdzona 27.11, odrzucona 28.11; kod w commicie 09c4a81,
  wycofany w następnym): etap 1 -- zdarzenia z typem, triggerem, subject, treatment,
  effect; etap 2 -- jedno wywołanie na zdarzenie: zdanie + zdarzenie -> 12 podargumentów.
  Gorsza od promptu jednoetapowego na dev i teście: 5-shot dev wszystkie argumenty 68,3
  vs 69,9 (autorzy 70,6), podargumenty 71,9 vs 74,4, rdzeń 80,2 vs 82,3, argumenty w
  zdarzeniu 51,4 vs 53,8; 0-shot 55,8 vs 61,9; tokeny promptu na zdanie +44% (4390 vs 3053).
  Traci głównie Treatment.Drug i Treatment.Disorder (choroba leży w 95% poza fragmentem
  leczenia), a etap 1 bez pól podargumentów gorzej wyznacza też argumenty główne. Zgodne z
  Sun i in. (2024): pipeline 0-shot gorszy niż najlepszy prompt jednokrokowy; my
  sprawdziliśmy wariant few-shot z jednym pytaniem na zdarzenie -- też gorszy.
- LLM (decyzja 27.11): w naszym prompcie definicje time elapsed i duration wymagają
  jawnej ilości czasu (bez samego „after”, „recently”, „following”), population wyklucza
  liczebniki porządkowe i przedimki. Powód: test 24.11, 5-shot, dev -- nasz prompt trafiał
  więcej poprawnych argumentów niż prompt autorów (293 vs 266 trafionych tylko przez jeden),
  ale miał 758 vs 576 zbędnych, głównie time elapsed (90 vs 16) i population (52 vs 8).
  Adnotacja: 95% time elapsed i 98% duration podaje ilość czasu. Argument do pracy: z 5
  przykładów model nie wyczyta takich reguł, więc są zapisane wprost w prompcie.
- Generative QA (27.11): bez zmian. Wersja rozwinięta wypisuje Subject w 483 zdaniach dev
  (prawdziwych 495), baseline w 544 -- baseline trafia więcej, bo strzela częściej. Model
  dostrojony uczy się konwencji z 2898 zdań treningowych, więc instrukcje w wejściu nie
  mają tu sensu; rozstrzygną seedy 2 i 3.
- Generative QA (decyzja 24.11): wersja rozwinięta `own_current_combined` -- etap 2
  trenowany na wszystkich pytaniach z prawdziwych argumentów głównych + pytaniach o
  niedokładne predykcje etapu 1 + pytaniach o predykcje bez dopasowania (odpowiedź
  „none”, żeby nie dopisywać podargumentów do nieistniejącego argumentu). Powód: w teście
  22.11 trening tylko na predykcjach obniżył pełność podargumentów (dev 84 -> 78) przy
  wyższej precyzji. Spadek był na prawie wszystkich etykietach, więc przyczyną mogą być też
  szablony lub format -- rozstrzygają warianty `own_current_gold` / `own_authors_predicted`.
- SL (decyzja 24.11): taggery argumentów trenowane na triggerach przewidzianych przez
  etap 1 na zbiorze treningowym (`train_inputs: predicted`) i co najmniej jeden trigger w
  zdaniu (`min_triggers: 1`). Powód: w teście 22.11 wersja z [TRG] trenowana na złotych
  triggerach miała niższą pełność (Treatment.Drug 90 -> 77, Treatment 69 -> 58) przy tej
  samej precyzji. Predykcje na treningu pochodzą z modelu trenowanego na tych zdaniach,
  więc są lepsze niż na teście (to samo ograniczenie co w QA).
- SL (decyzja 21.11): bez łączonych tagów dla fragmentów z dwiema etykietami
  (Subject.Disorder = Treatment.Disorder, Duration = Time_elapsed). Słowo ma jeden tag,
  wygrywa częstsza etykieta (u autorów: ostatnia w ich kolejności, ten sam wynik), więc
  przegrana etykieta znika w treningu. Opisać w pracy jako ograniczenie SL (QA i LLM go
  nie mają: osobne pytania / pola). Sufit [TRG]: argumenty w zdarzeniu 95,2 ogółem.
- Później: analiza błędów (granica / zła etykieta / pominięty / nadmiarowy).

Stan na 28.10.2026, koniec sesji implementacji (przerwanej z powodu limitu kredytów).

## Do decyzji

(brak)

Rozstrzygnięte:
- SL i extractive QA: BioClinical ModernBERT (według autorów lepszy niż Bio-ModernBERT).
- Generative QA: zostaje T5Gemma (decyzja z 29.10).
- LLM: zostają Qwen3-8B i Llama-3.1-8B.
- Plan eksperymentów (3.11): dwie wersje na metodę, baseline autorów i wersja poprawiona
  (SL `own_crf`, extractive QA i generative QA `own_current_predicted`, LLM Qwen3-8B 0/5-shot);
  najpierw tryb testowy na seedzie 1, potem seedy 2 i 3. Reszta wariantów w grupach `extra_*`.
- Extractive QA (1.11): baseline autorów z taggerem triggerów jak w EEQA; własny pipeline
  z pytaniem o trigger, które zwraca wszystkie odpowiedzi powyżej „brak odpowiedzi”
  (min. 1); limit 2 odpowiedzi na pytanie wszędzie, jak w kodzie autorów.
- Extractive QA (decyzja 24.11): baseline z regułą wyboru odpowiedzi 1:1 z kodu autorów
  (`selection: authors`). Tryb testowy 22.11 pokazał precyzję argumentów głównych 45
  (pełność 68) i 54,4 EM-F1 wobec 65,7 w artykule; ich EXP_RECORD.md: ta sama precyzja
  przy pierwotnej regule EEQA, 65,7 po zmianie reguły.
- Extractive QA, wersja rozwinięta (decyzja 24.11): ten sam odczyt odpowiedzi co
  baseline (reguła autorów, maks. 30 tokenów; wcześniej 60/50 i stara reguła, precyzja
  argumentów 44, triggerów 44). Różni się tylko trzema zmianami: triggery z klasyfikatora
  typu i pytania o trigger (reguła autorów + min. 1 odpowiedź), własne szablony, trening
  etapów argumentów na predykcjach.

## Niesprawdzone lokalnie (brak GPU i dostępu do Hugging Face w środowisku)

3. Prawdziwe modele i tokenizery (BioBERT, BioClinical ModernBERT, SciFive, T5Gemma):
   sprawdzi je `bash scripts/vast/quick_check.sh` (pierwszy krok na vast.ai).
4. bf16 na GPU, pomiar pamięci GPU, czasy: również quick_check.
5. **Część LLM (vLLM) w ogóle nie była uruchomiona** – lokalnie tylko backend `hf` na
   małym losowym modelu. Na vast.ai najpierw krótki test:
   `PYTHONPATH=$PWD /workspace/venv-llm/bin/python scripts/run_experiment.py --config configs/llm/qwen3_8b.yaml --output_dir results_quick --set data.limit=20 --force`
6. ModernBERT: wbudowany `torch.compile` wyłączony (`model.compile: false`), bo wymaga
   kompilatora C na maszynie; można włączyć, jeśli quick_check pokaże, że działa i
   przyspiesza.

## Niedokończone weryfikacje

7. Niezależny przegląd kodu (osobny agent) został przerwany przed końcem; jego
   uwagi i stan poprawek są poniżej.
8. Pełny smoke test po ostatnich zmianach przeszedł we wszystkich 16 konfiguracjach
   (CPU, małe losowe modele). Powtórka: `python scripts/smoke/make_tiny_models.py &&
   python scripts/smoke/run_smoke.py`.

## Koszt

9. Szacunek (3.11) dla planu głównego: tryb testowy ok. 4 h na RTX 4090 / 6 h na 3090,
   seedy 2 i 3 ok. 6 h / 8–9 h; najdłużej generative QA. Dokładne liczby da quick_check.

## Uwagi z przeglądu kodu

Przegląd przerwany przed końcem; nic krytycznego. Wszystkie zweryfikowane uwagi są
**poprawione**:

- Extractive QA: pytanie o trigger może zwrócić kilka odpowiedzi (powyżej „brak
  odpowiedzi”), więc zdanie z dwoma zdarzeniami tego samego typu da się w pełni
  rozwiązać (w teście 17 takich zdań).
- SL: tokeny złożone z samych spacji usunięte przy konwersji danych (6 w całym zbiorze).
- Generative QA: niedokończony znacznik na końcu uciętego wyjścia jest odrzucany, a
  liczba uciętych wyjść zapisywana.
- Pamięć GPU dla vLLM: raportowany rozmiar wag zamiast pamięci zarezerwowanej.

Sprawdzone i poprawne: normalizacja identyczna z oficjalnym ewaluatorem autorów,
konwersja danych (bez przecieków między train/dev/test), pozycje odpowiedzi w QA dla obu
typów tokenizerów, early stopping z przywracaniem najlepszych wag, CRF i Viterbi w fp32,
token startowy T5Gemma, dekodowanie ograniczone, czas BM25 (ok. 6 ms na zapytanie).
