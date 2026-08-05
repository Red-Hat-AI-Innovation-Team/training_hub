"""Backend adapters for TrainingHubCallback.

Each adapter translates TrainingHubCallback lifecycle hooks into a
backend's native callback interface:

- ``unsloth``: in-process HuggingFace TrainerCallback wrap
- ``instructlab`` / ``mini_trainer``: torchrun-safe bridges + payload file
"""
