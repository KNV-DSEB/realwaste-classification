from src.models.efficientnet import EfficientNetB0Transfer
from src.models.multiscale_cnn import MultiScaleCNN
from src.models.simple_cnn import SimpleCNN

MODELS = {"SimpleCNN": SimpleCNN, "MultiScaleCNN": MultiScaleCNN, "EfficientNet-B0": EfficientNetB0Transfer}


def build_model(model_name, **kwargs):
    # checkpoints store model_name + kwargs, so any saved model can be rebuilt with this
    return MODELS[model_name](**kwargs)
