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
