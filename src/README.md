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

Planned:
- `train.py` — common training loop/checkpoint logic.
- `evaluate.py` — one common metrics implementation.
- `models/simple_cnn.py`
- `models/multiscale_cnn.py`
- `models/efficientnet.py`

Do not duplicate the entire training/evaluation pipeline in each notebook.
