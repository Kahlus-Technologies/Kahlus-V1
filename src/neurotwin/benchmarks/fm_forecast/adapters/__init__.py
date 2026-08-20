"""Per-model adapters wrapping vendored pretrained EEG encoders as forecast runners.

Each module exposes the same surface (MODEL_ID, SR_OUT, N_SAMPLES_OUT, EMBED_DIM,
load_encoder(), make_runner()) and is imported lazily: importing this package must
never import torch, vendored upstream code, or trigger checkpoint IO. Import the
concrete adapter module (e.g. ``adapters.eegpt_adapter``) only inside the runner
that needs it.
"""
