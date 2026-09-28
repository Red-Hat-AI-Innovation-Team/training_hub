import logging as _logging
import os as _os


def _configure_package_logging() -> None:
    """Make training_hub's own INFO records visible in an unconfigured process.

    Training runs in pods and scripts where nothing calls ``basicConfig``, and
    Python's default WARNING gate would drop every INFO record -- including the
    resume decision, which an operator needs to tell a resumed run from a
    silent restart at step 0. Configuring the ``training_hub`` logger (never the
    root logger) keeps that visible without speaking for the rest of the
    process. An application that configured logging itself wins: we only act
    when neither our logger nor the root has a handler.

    ``TRAINING_HUB_LOG_LEVEL`` overrides the level; set it to a higher level to
    quiet these records.
    """
    logger = _logging.getLogger("training_hub")
    level = _os.environ.get("TRAINING_HUB_LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    if logger.handlers or _logging.root.handlers:
        return
    handler = _logging.StreamHandler()
    handler.setFormatter(_logging.Formatter("[%(name)s|%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    # Stop propagating once we own a handler. The training backends call
    # basicConfig() from inside run_training, long after this import, and a
    # record reaching both their root handler and ours prints every checkpoint
    # line twice in the pod log.
    logger.propagate = False


_configure_package_logging()

from .algorithms import Algorithm, Backend, AlgorithmRegistry, create_algorithm
from .algorithms.sft import sft, SFTAlgorithm, InstructLabTrainingSFTBackend
from .algorithms.osft import OSFTAlgorithm, MiniTrainerOSFTBackend, osft
from .algorithms.lora import lora_sft, LoRASFTAlgorithm, UnslothLoRABackend
from .algorithms.lora_grpo import lora_grpo, grpo, LoRAGRPOAlgorithm, ARTLoRAGRPOBackend
from .algorithms.lora_grpo_verl import VeRLLoRAGRPOBackend
from .algorithms.gepa import gepa, GEPAAlgorithm, GEPABackend, MLflowGEPABackend
from .algorithms.rewards import tool_call_reward, binary_reward
from .callbacks import TrainingHubCallback, TrainingHubContext, TrainingHubControl, merge_default_callbacks
from .jit_checkpoint import JITCheckpointCallback, RemoteCheckpointSyncCallback
from .hub_core import welcome
from .profiling.memory_estimator import BasicEstimator, OSFTEstimatorExperimental, estimate, OSFTEstimator, LoRAEstimator, QLoRAEstimator
from .algorithms.its_rollout import ITSRollout
from .visualization import plot_loss

__all__ = [
    'Algorithm',
    'Backend',
    'AlgorithmRegistry',
    'create_algorithm',
    'sft',
    'osft',
    'lora_sft',
    'lora_grpo',
    'grpo',
    'SFTAlgorithm',
    'InstructLabTrainingSFTBackend',
    'OSFTAlgorithm',
    'MiniTrainerOSFTBackend',
    'LoRASFTAlgorithm',
    'UnslothLoRABackend',
    'LoRAGRPOAlgorithm',
    'ARTLoRAGRPOBackend',
    'VeRLLoRAGRPOBackend',
    'gepa',
    'GEPAAlgorithm',
    'GEPABackend',
    'MLflowGEPABackend',
    'tool_call_reward',
    'binary_reward',
    'welcome',
    'BasicEstimator',
    'OSFTEstimatorExperimental',
    'OSFTEstimator',
    'LoRAEstimator',
    'QLoRAEstimator',
    'estimate',
    'TrainingHubCallback',
    'TrainingHubContext',
    'TrainingHubControl',
    'merge_default_callbacks',
    'JITCheckpointCallback',
    'RemoteCheckpointSyncCallback',
    'ITSRollout',
    'plot_loss',
]
