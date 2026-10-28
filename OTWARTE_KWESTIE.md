# Otwarte kwestie (do sprawdzenia później)

Stan na 28.10.2026, koniec sesji implementacji (przerwanej z powodu limitu kredytów).

## Do decyzji

1. **Model enkodera w SL i extractive QA.** W planie był Bio-ModernBERT. Zmieniłem na
   `thomas-sounack/BioClinical-ModernBERT-base`, bo autorzy obu modeli piszą (karta modelu
   i arXiv 2506.10896), że Bio-ModernBERT konsekwentnie wypadał gorzej. Powrót: jedna
   linijka `model.name` w `configs/seq/own_base.yaml` i `configs/eqa/own_base.yaml`.
   Jeśli zostaje BioClinical, trzeba też poprawić nazwę modelu w dokumencie z planem.

## Niesprawdzone lokalnie (brak GPU i dostępu do Hugging Face w środowisku)

2. Prawdziwe modele i tokenizery (BioBERT, BioClinical ModernBERT, SciFive, T5Gemma):
   sprawdzi je `bash scripts/vast/quick_check.sh` (pierwszy krok na vast.ai).
3. bf16 na GPU, pomiar pamięci GPU, czasy: również quick_check.
4. **Część LLM (vLLM) w ogóle nie była uruchomiona** – lokalnie tylko backend `hf` na
   małym losowym modelu. Na vast.ai najpierw krótki test:
   `PYTHONPATH=$PWD /workspace/venv-llm/bin/python scripts/run_experiment.py --config configs/llm/qwen3_8b.yaml --output_dir results_quick --set data.limit=20 --force`
5. ModernBERT: wbudowany `torch.compile` wyłączony (`model.compile: false`), bo wymaga
   kompilatora C na maszynie; można włączyć, jeśli quick_check pokaże, że działa i
   przyspiesza.

## Niedokończone weryfikacje

6. Niezależny przegląd kodu (osobny agent) został przerwany przed końcem; jego
   dotychczasowe uwagi są poniżej (jeśli zdążył je zwrócić).
7. Pełny smoke test po ostatnich zmianach (ładowanie modeli, LLM, tabele) był w trakcie;
   wcześniejsza wersja przeszła wszystkie 16 konfiguracji. Przed startem na vast.ai:
   `python scripts/smoke/make_tiny_models.py && python scripts/smoke/run_smoke.py`
   (CPU, kilka minut).

## Koszt

8. Najdłużej potrwa generative QA (etap 2: ok. 36 tys. pytań o subargumenty na epokę,
   generowanie beam 3 na dev po każdej epoce). Szacunek bardzo zgrubny: kilkanaście godzin
   GPU dla całej macierzy; dokładne liczby da quick_check. Opcja oszczędności:
   `--set early_stopping_dev_limit=300`.

## Uwagi z przeglądu kodu

(brak – uzupełnić)
