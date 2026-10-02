# src

Shared code goes here after Gate 1/2 approval.

Planned modules:
- `dataset.py` — read frozen split.csv and build datasets/dataloaders.
- `transforms.py` — shared train vs validation/test transforms.
- `train.py` — common training loop/checkpoint logic.
- `evaluate.py` — one common metrics implementation.
- `utils.py` — seeds, paths, logging.
- `models/simple_cnn.py`
- `models/multiscale_cnn.py`
- `models/efficientnet.py`

Do not duplicate the entire training/evaluation pipeline in each notebook.
