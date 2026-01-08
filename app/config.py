import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    # MODEL PATH 
    MODEL_PATH = os.path.join(
        BASE_DIR, "..", "models", "skin_disease_resnet152v2.keras"
    )

    CLASS_NAMES_PATH = os.path.join(
        BASE_DIR, "..", "models", "class_names.json"
    )
