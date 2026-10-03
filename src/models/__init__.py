"""Model registry: every checkpoint stores `model_name` and `model_kwargs` in its `run_info`, so the
final-evaluation notebook can rebuild any experiment's model with `build_model`."""
from src.models.multiscale_cnn import MultiScaleCNN
from src.models.simple_cnn import SimpleCNN

MODELS = {
    "SimpleCNN": SimpleCNN,
    "MultiScaleCNN": MultiScaleCNN,
}


def build_model(model_name, **model_kwargs):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}; known: {sorted(MODELS)}")
    return MODELS[model_name](**model_kwargs)
