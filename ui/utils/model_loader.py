import joblib
from pathlib import Path

class ModelLoader:
    def __init__(self, models_dir="models/saved"):
        self.models_dir = Path(models_dir)

    def _load(self, name):
        path = self.models_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        return joblib.load(path)

    @property
    def xgboost(self):
        return self._load("xgboost.pkl")

    @property
    def logreg(self):
        return self._load("logistic_regression.pkl")

    @property
    def scaler(self):
        return self._load("scaler.pkl")

    @property
    def encoders(self):
        return self._load("label_encoders.pkl")

def get_model_loader():
    return ModelLoader()
