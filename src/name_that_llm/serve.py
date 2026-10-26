"""Production wiring: the app the Docker image runs (`uvicorn name_that_llm.serve:app`).

api.py builds the web app around ANY predictor; this file hands it the real one: the
registered champion, loaded from the MLflow registry (registry.load_registered). Importing
this module loads nothing: the model loads in the background once the app starts.

Settings, from the environment (the Space's variables and secrets):
  MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD   where the registry is, and the DagsHub token
  MODEL_NAME, MODEL_ALIAS                       default "name-that-llm" and "champion"
  ENCODER_FOLDS                                 fewer encoder fold models (the fallback)
  TORCH_THREADS                                 default 2: the free Space's 2 vCPUs
"""

import os

from name_that_llm.api import create_app


def _load():
    import torch

    from name_that_llm.registry import load_registered

    # Match the Space's 2 vCPUs: more threads than cores only adds switching overhead.
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    predictor, _ = load_registered(
        os.environ.get("MODEL_NAME", "name-that-llm"), os.environ.get("MODEL_ALIAS", "champion")
    )
    return predictor


app = create_app(_load)
