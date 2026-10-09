# src

Shared code; the notebooks only call it.

| File | What it does |
|---|---|
| `utils.py` | config, Drive paths, SHA256, seeds |
| `split.py` | grouped split (notebook 02b) |
| `duplicate_check.py` | exact/near duplicates (notebook 03) |
| `transforms.py` | train augmentation, eval resize + normalize |
| `dataset.py` | frozen split → Dataset → DataLoaders; `get_test_loader` only for notebook 09 |
| `models/` | E1 `SimpleCNN`, E2 `MultiScaleCNN`, E3 `EfficientNetB0Transfer`; `build_model(name, **kwargs)` |
| `train.py` | training loop: early stopping on val macro-F1, checkpoint every epoch, resume |
| `evaluate.py` | predictions, metrics, confusion matrix |
| `tuning.py` | extra runs (now only seeds 43/44, D030), one folder per run (notebook 10) |
| `final_eval.py` | test evaluation, once per checkpoint (notebook 09) |
