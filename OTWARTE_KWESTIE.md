# Otwarte kwestie (do sprawdzenia później)

Stan na 28.10.2026, koniec sesji implementacji (przerwanej z powodu limitu kredytów).

## Do decyzji

1. **Extractive QA, kilka triggerów tego samego typu.** Na branchu: najlepsza odpowiedź
   + kolejne w granicy marginesu wybieranego na dev. Do wyboru:
   A) jedna odpowiedź na pytanie o trigger (pierwotna konstrukcja) i opis ograniczenia
      (ok. 13 z 1010 zdarzeń w teście nieosiągalnych),
   B) pytanie o trigger działa jak pytania o argumenty: wszystkie odpowiedzi powyżej
      „brak odpowiedzi” (min. 1, maks. 3); w treningu dochodzą pytania o typ zdarzenia,
      którego w zdaniu nie ma,
   C) obecna wersja z marginesem.

Rozstrzygnięte:
- SL i extractive QA: BioClinical ModernBERT (według autorów lepszy niż Bio-ModernBERT).
- Generative QA: zostaje T5Gemma (decyzja z 29.10).
- LLM: zostają Qwen3-8B i Llama-3.1-8B.

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

9. Najdłużej potrwa generative QA (etap 2: ok. 36 tys. pytań o subargumenty na epokę,
   generowanie beam 3 na dev po każdej epoce). Szacunek bardzo zgrubny: kilkanaście godzin
   GPU dla całej macierzy; dokładne liczby da quick_check. Opcja oszczędności:
   `--set early_stopping_dev_limit=300`.

## Uwagi z przeglądu kodu

Przegląd przerwany przed końcem; nic krytycznego. Wszystkie zweryfikowane uwagi są
**poprawione**:

- Extractive QA: pytanie o trigger może zwrócić kilka odpowiedzi (margines wybierany
  na dev), więc zdanie z dwoma zdarzeniami tego samego typu da się w pełni rozwiązać
  (w teście 17 takich zdań).
- SL: tokeny złożone z samych spacji usunięte przy konwersji danych (6 w całym zbiorze).
- Generative QA: niedokończony znacznik na końcu uciętego wyjścia jest odrzucany, a
  liczba uciętych wyjść zapisywana.
- Pamięć GPU dla vLLM: raportowany rozmiar wag zamiast pamięci zarezerwowanej.

Sprawdzone i poprawne: normalizacja identyczna z oficjalnym ewaluatorem autorów,
konwersja danych (bez przecieków między train/dev/test), pozycje odpowiedzi w QA dla obu
typów tokenizerów, early stopping z przywracaniem najlepszych wag, CRF i Viterbi w fp32,
token startowy T5Gemma, dekodowanie ograniczone, czas BM25 (ok. 6 ms na zapytanie).
