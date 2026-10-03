# src

Shared code for every experiment. Notebooks import it from a clone of this repo
(`sys.path.insert(0, REPO_DIR)`); do not copy pipeline code into notebooks.

Implemented:
- `utils.py` — config loading, Drive paths, SHA256, seeds, environment record.
- `transforms.py` — shared train (augmented) vs validation/test (deterministic) transforms.
- `dataset.py` — SHA256-verified frozen split → `RealWasteDataset` → DataLoaders.
  `get_dataloaders` returns train/val only; `get_test_loader` is for the final evaluation only.
- `duplicate_check.py` — exact/near-duplicate detection used by `notebooks/03_duplicate_check.ipynb`.
- `split.py` — grouped stratified split (blocks of consecutive file numbers) used by `notebooks/02b_grouped_split.ipynb` (D015).
- `train.py` — shared training loop: validation-macro-F1 selection and early stopping, Drive checkpoints every epoch, resume after a disconnect, training curves.
- `evaluate.py` — one metrics implementation (accuracy, macro P/R/F1, per-class, confusion matrix) for validation and the final test evaluation.
- `models/simple_cnn.py` — E1.
- `models/multiscale_cnn.py` — E2.
- `models/efficientnet.py` — E3: ImageNet EfficientNet-B0 with a new head; `set_trainable_blocks(n)` freezes all but the last `n` feature blocks and keeps frozen BatchNorm layers in eval mode.
- `models/__init__.py` — `build_model(model_name, **model_kwargs)`, used to rebuild any model from the `run_info` stored in its checkpoints.

Do not duplicate the entire training/evaluation pipeline in each notebook.
