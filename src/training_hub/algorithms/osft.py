import os
from typing import Literal, get_origin, get_args, Union
from dataclasses import fields
import warnings

import datasets
from training_hub.algorithms import Algorithm, Backend, AlgorithmRegistry
from training_hub.utils import format_type_name, get_torchrun_params


class OSFTAlgorithm(Algorithm):
    """
    Implements the Orthogonal Subspace Fine-Tuning (OSFT) algorithm,
    based on Nayak et al. (2025), arXiv:2504.07097

    This algorithm allows for continual training of pre-trained or instruction-tuned
    models without the need of a supplementary dataset to maintain the distribution
    of the original model/dataset that was trained.
    """

    def __init__(self, backend: Backend, **kwargs) -> None:
        self.backend = backend
        self.kwargs = kwargs

    def train(
        self,
        model_path: str,
        data_path: str,
        unfreeze_rank_ratio: float,
        effective_batch_size: int,
        max_tokens_per_gpu: int,
        max_seq_len: int,
        learning_rate: float,
        ckpt_output_dir: str,
        # patterns that we want to match against when selecting
        # modules for OSFT
        target_patterns: list[str] | None = None,
        # settings for training mode
        seed: int | None = None,
        use_liger: bool | None = None,
        # learning rate scheduler
        lr_scheduler: str = None,
        warmup_steps: int = None,
        lr_scheduler_kwargs: dict[str, str] | None = None,
        # AdamW optimizer parameters
        beta1: float | None = None,
        beta2: float | None = None,
        eps: float | None = None,
        weight_decay: float | None = None,
        # checkpointing
        checkpoint_at_epoch: bool | None = None,
        save_final_checkpoint: bool | None = None,
        # parameters for the training mode
        num_epochs: int | None = None,
        # whether to use the processed dataset
        use_processed_dataset: bool | None = None,
        unmask_messages: bool | None = None,
        data_output_dir: str | None = None,
        # Pretraining mode parameters
        is_pretraining: bool | None = None,
        block_size: int | None = None,
        document_column_name: str | None = None,
        # Torchrun parameters for multi-node support
        nproc_per_node: Literal['auto', 'gpu'] | int | None = None,
        nnodes: int | None = None,
        node_rank: int | None = None,
        rdzv_id: str | int | None = None,
        rdzv_endpoint: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        # Logging parameters
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        tensorboard_log_dir: str | None = None,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
        mlflow_run_name: str | None = None,
        **kwargs,
    ) -> any:
        """
        This algorithm implements Continual Training using the OSFT algorithm
        with the mini-trainer backend.

        **Note:** The OSFT algorithm does not reduce the memory requirement when compared,
        to SFT, but it significantly reduces the data requirement for customizing an instruction-tuned
        model compared to SFT.

        **Note:**
            While all values of `unfreeze_rank_ratio` are valid, in practice you will seldom
            need values greater than 0.5 for general continual-learning regimes.

        Arguments:
            model_path (str): Local path or HuggingFace model ID to be used for fine-tuning.
            data_path (str):
                Path to the training data. When `use_processed_dataset` is True,
                this is the path to the processed dataset. When `use_processed_dataset` is False,
                this is the path to the original dataset.
            unfreeze_rank_ratio (float):
                Controls the amount that each matrix is unfrozen during OSFT.
                Valid values are between 0.0 and 1.0.
            effective_batch_size (int): Effective batch size for training.
            max_tokens_per_gpu (int):
                The maximum number of tokens placed on a single GPU for training.
                When hitting OOMs, consider reducing this value.
            max_seq_len (int):
                Sets the maximum sequence length (in tokens) of samples that will be used for training.
                Any sample exceeding this length will be dropped from the dataset.
            learning_rate (float): Learning rate for model update size.
            ckpt_output_dir (str):
                Directory where outputs from training will be saved such as checkpoints and logs.
                any necessary intermediate files.
            target_patterns (list[str]):
                List of patterns to match against when selecting modules for OSFT,
                useful for custom training regimes or enabling OSFT for custom models which
                do not have pre-defined defaults.
            seed (int): Random seed for training.
            use_liger (bool): Whether to use Liger kernels for training.
            lr_scheduler (str): Name of the PyTorchlearning rate scheduler to use.
            warmup_steps (int): Number of warmup steps for the learning rate scheduler.
            lr_scheduler_kwargs (dict[str, str]): Additional scheduler parameters.
            beta1 (float): AdamW optimizer beta1 coefficient (momentum).
            beta2 (float): AdamW optimizer beta2 coefficient (RMSprop).
            eps (float): AdamW optimizer epsilon for numerical stability.
            weight_decay (float): AdamW optimizer weight decay coefficient.
            checkpoint_at_epoch (bool): Whether to checkpoint at each epoch.
            save_final_checkpoint (bool): Whether to save final checkpoint once training is complete.
            num_epochs (int): Number of epochs to train for.
            use_processed_dataset (bool):
                Whether to use the processed dataset. If False, the data is assumed to be in standard
                messages format witha `messages` and optional `unmask` field on each sample.
                When True, we assume that each sample has an `input_ids` and `labels` field containing
                data tokenized for the model being trained.
            unmask_messages (bool):
                Whether to unmask messages during data processing. This value is ignored
                when `use_processed_dataset` is True.
            data_output_dir (str):
                Directory where outputs from data processing will be saved such as intermediate
                files. When not provided, it defaults to `_internal_data_processing` under the
                `ckpt_output_dir`.
            is_pretraining (bool):
                Enable pretraining mode. Expects data with {"documents": "text"} format.
                Data is tokenized without chat templates. Blocking happens in mini-trainer.
                Mutually exclusive with unmask_messages.
            block_size (int | None):
                Block size in tokens for pretraining. Required when is_pretraining=True.
                Passed to mini-trainer for block-based sampling.
            document_column_name (str | None):
                Column name containing raw documents when `is_pretraining=True`.
                Defaults to "document" when not provided.
            nproc_per_node (Literal['auto', 'gpu'] | int): Number of processes (GPUs) per node for distributed training.
            nnodes (int): Total number of nodes for distributed training.
            node_rank (int): Rank of this node (0 to nnodes-1) for distributed training.
            rdzv_id (str | int): Unique job ID for rendezvous in distributed training.
            rdzv_endpoint (str): Master node endpoint for multi-node training.
            master_addr (str): Master node address for distributed training (only used with
                static rdzv_backend).
            master_port (int): Master node port for distributed training.
            wandb_project (str): Weights & Biases project name.
            wandb_entity (str): Weights & Biases team/entity name.
            wandb_run_name (str): Weights & Biases run name.
            tensorboard_log_dir (str): Directory for TensorBoard logs.
            mlflow_tracking_uri (str): MLflow tracking server URI.
            mlflow_experiment_name (str): MLflow experiment name.
            mlflow_run_name (str): MLflow run name.
            **kwargs: Additional parameters passed to the backend.

        Returns:
            None
        """

        # param validation
        if not (0.0 <= unfreeze_rank_ratio <= 1.0):
            raise ValueError(f'unfreeze_rank_ratio must be between 0.0 and 1.0, but got {unfreeze_rank_ratio}')

        if is_pretraining and block_size is None:
            raise ValueError('block_size required when is_pretraining=True')

        # Validate pretraining parameters
        if is_pretraining and unmask_messages:
            raise ValueError(
                'Cannot use both is_pretraining=True and unmask_messages=True. These are mutually exclusive modes.'
            )

        if not is_pretraining and block_size is not None:
            warnings.warn('block_size only valid with is_pretraining=True', stacklevel=2)

        required_params = {
            'model_path': model_path,
            'data_path': data_path,
            'effective_batch_size': effective_batch_size,
            'max_tokens_per_gpu': max_tokens_per_gpu,
            'max_seq_len': max_seq_len,
            'learning_rate': learning_rate,
            'ckpt_output_dir': ckpt_output_dir,
            'unfreeze_rank_ratio': unfreeze_rank_ratio,
        }

        optional_params = {
            'target_patterns': target_patterns,
            # for data processing
            'use_processed_dataset': use_processed_dataset,
            'unmask_messages': unmask_messages,
            'data_output_dir': data_output_dir,
            # pretraining params
            'is_pretraining': is_pretraining,
            'block_size': block_size,
            'document_column_name': document_column_name,
            # scheduler params
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'warmup_steps': warmup_steps,
            # AdamW optimizer parameters
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'weight_decay': weight_decay,
            # checkpointing settings
            'checkpoint_at_epoch': checkpoint_at_epoch,
            'save_final_checkpoint': save_final_checkpoint,
            'num_epochs': num_epochs,
            'use_liger': use_liger,
            'seed': seed,
            # torchrun params
            'nproc_per_node': nproc_per_node,
            'nnodes': nnodes,
            'node_rank': node_rank,
            'rdzv_id': rdzv_id,
            'rdzv_endpoint': rdzv_endpoint,
            'master_addr': master_addr,
            'master_port': master_port,
            # logging params
            'wandb_project': wandb_project,
            'wandb_entity': wandb_entity,
            'wandb_run_name': wandb_run_name,
            'tensorboard_log_dir': tensorboard_log_dir,
            'mlflow_tracking_uri': mlflow_tracking_uri,
            'mlflow_experiment_name': mlflow_experiment_name,
            'mlflow_run_name': mlflow_run_name,
        }

        # now do validation now that we've set everything up
        for required_param in self.get_required_params().keys():
            if required_param not in required_params:
                raise ValueError(f'error: required parameter not provided: {required_param}')

        # validate types of all parameters
        self._validate_param_types(required_params)
        self._validate_param_types(optional_params)
        self._validate_param_types(kwargs)

        all_params = dict(**required_params)
        all_params.update(optional_params)
        all_params.update(kwargs)

        return self.backend.execute_training(all_params)

    def get_required_params(self) -> dict[str, type]:
        """Return dictionary of required parameter names and their types."""
        return {
            'model_path': str,
            'data_path': str,
            'unfreeze_rank_ratio': float,
            'effective_batch_size': int,
            'max_tokens_per_gpu': int,
            'max_seq_len': int,
            'learning_rate': float,
            'ckpt_output_dir': str,
        }

    def get_optional_params(self) -> dict[str, type]:
        """Return dictionary of optional parameter names and their types."""
        return {
            'target_patterns': list[str],
            'seed': int,
            'use_liger': bool,
            'lr_scheduler': str,
            'warmup_steps': int,
            'lr_scheduler_kwargs': dict,
            # AdamW optimizer parameters
            'beta1': float,
            'beta2': float,
            'eps': float,
            'weight_decay': float,
            'checkpoint_at_epoch': bool,
            'save_final_checkpoint': bool,
            'num_epochs': int,
            'use_processed_dataset': bool,
            'unmask_messages': bool,
            'data_output_dir': str,
            'is_pretraining': bool,
            'block_size': int,
            'document_column_name': str,
            'nproc_per_node': Literal['auto', 'gpu'] | int,
            'nnodes': int,
            'node_rank': int,
            'rdzv_id': str | int,
            'rdzv_endpoint': str,
            'master_addr': str,
            'master_port': int,
            # logging params
            'wandb_project': str,
            'wandb_entity': str,
            'wandb_run_name': str,
            'tensorboard_log_dir': str,
            'mlflow_tracking_uri': str,
            'mlflow_experiment_name': str,
            'mlflow_run_name': str,
        }

    def _validate_param_types(self, params: dict[str, any]):
        """Type-check given parameters, handling modern Python typing constructs."""
        required_param_types = self.get_required_params()
        optional_param_types = self.get_optional_params()
        all_param_types = {**required_param_types, **optional_param_types}

        for param, value in params.items():
            # use 'any' here to handle the case when the param is not defined by
            # either optional or required
            param_type = all_param_types.get(param, any)

            # allow optional params to be None
            if param in optional_param_types and value is None:
                continue  # None is allowed for optional params

            if not self._check_type(value, param_type):
                err_msg = (
                    f"error: param '{param}' received unexpected type, "
                    f"expected '{format_type_name(param_type)}' but got '{format_type_name(type(value))}'"
                )
                raise ValueError(err_msg)

    def _check_type(self, value, expected_type) -> bool:
        """Check if value matches expected_type, handling modern typing constructs."""
        # Handle 'any' type (accepts anything)
        if expected_type is any:
            return True

        # Handle basic types that work with isinstance
        try:
            if isinstance(expected_type, type):
                return isinstance(value, expected_type)
        except TypeError:
            pass  # Fall through to handle complex types

        # Handle parameterized generics and unions
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle Union types (including X | None syntax)
        if origin is Union:
            return any(self._check_type(value, arg) for arg in args)

        # Handle list types
        if origin is list:
            if not isinstance(value, list):
                return False
            if args and value:  # Check element types if specified and list is not empty
                element_type = args[0]
                return all(self._check_type(item, element_type) for item in value)
            return True

        # Handle dict types
        if origin is dict:
            if not isinstance(value, dict):
                return False
            if args and value:  # Check key/value types if specified and dict is not empty
                key_type, val_type = args[0], args[1]
                return all(self._check_type(k, key_type) and self._check_type(v, val_type) for k, v in value.items())
            return True

        # Fallback for basic isinstance check
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # If we can't check the type, assume it's valid
            return True


class MiniTrainerOSFTBackend(Backend):
    """MiniTrainer backend for OSFT algorithm."""

    def execute_training(self, algorithm_params: dict[str, any]) -> any:
        """
        Execute OSFT training using MiniTrainer.

        Since this backend doesn't do its own data processing, it delegates that to the instructlab-training
        backend.


        """
        from mini_trainer import (
            run_training,
            TrainingArgs,
            TorchrunArgs,
            TrainingMode,
            PretrainingConfig,
        )

        # here we translate the parameter names that the algorithm used
        # into those used by the backend
        renames = {
            'use_liger': 'use_liger_kernels',
            'warmup_steps': 'num_warmup_steps',
            'target_patterns': 'osft_target_patterns',
            'unfreeze_rank_ratio': 'osft_unfreeze_rank_ratio',
            'model_path': 'model_name_or_path',
            'num_epochs': 'max_epochs',
            'effective_batch_size': 'batch_size',
            'ckpt_output_dir': 'output_dir',
        }

        # Rename parameters before sending to backend
        algorithm_params = {renames.get(k, k): v for k, v in algorithm_params.items()}

        # Populate logging params from environment variables if not explicitly set
        if not algorithm_params.get('mlflow_tracking_uri'):
            algorithm_params['mlflow_tracking_uri'] = os.environ.get('MLFLOW_TRACKING_URI')
        if not algorithm_params.get('mlflow_experiment_name'):
            algorithm_params['mlflow_experiment_name'] = os.environ.get('MLFLOW_EXPERIMENT_NAME')
        if not algorithm_params.get('mlflow_run_name'):
            algorithm_params['mlflow_run_name'] = os.environ.get('MLFLOW_RUN_NAME')
        if not algorithm_params.get('wandb_project'):
            algorithm_params['wandb_project'] = os.environ.get('WANDB_PROJECT')
        if not algorithm_params.get('wandb_entity'):
            algorithm_params['wandb_entity'] = os.environ.get('WANDB_ENTITY')
        if not algorithm_params.get('wandb_run_name'):
            algorithm_params['wandb_run_name'] = os.environ.get('WANDB_RUN_NAME')

        # Separate parameters into their respective dataclass fields
        torchrun_args_fields = {f.name for f in fields(TorchrunArgs)}
        training_args_fields = {f.name for f in fields(TrainingArgs)}

        # process this up here so we can exit early
        torchrun_args_pre = {k: v for k, v in algorithm_params.items() if k in torchrun_args_fields and v is not None}
        torchrun_args_pre = get_torchrun_params(torchrun_args_pre)
        torch_args = TorchrunArgs(**torchrun_args_pre)

        # We separate this from `ckpt_output_dir` so that we can use `/dev/shm` for low-latency data
        # proceessing. But we do not want to make assumptions about the size of training data or the
        # amount of memory on the host. So by default we write to storage, but expose this as a separate
        # parameter for performaance gains.
        data_output_dir = algorithm_params.get('data_output_dir', None)
        if data_output_dir is None:
            data_output_dir = os.path.join(algorithm_params['output_dir'], '_internal_data_processing')

        # since mini trainer itself does not process data, we delegate this to
        # a separate backend, and expect to receive the correct data path
        training_ready_data_path = self._process_data(
            data_path=algorithm_params['data_path'],  # should be there
            model_name_or_path=algorithm_params['model_name_or_path'],  # should be there
            output_dir=data_output_dir,
            max_seq_len=algorithm_params['max_seq_len'],
            num_cpu_procs=8,  # this is a safe default
            use_processed_dataset=algorithm_params.get('use_processed_dataset', False),
            unmask_messages=algorithm_params.get('unmask_messages', False),
            is_pretraining=algorithm_params.get('is_pretraining', False),
            document_column_name=algorithm_params.get('document_column_name'),
        )

        # adjust arguments to align with the API definition
        training_args_pre = {k: v for k, v in algorithm_params.items() if k in training_args_fields and v is not None}
        training_args_pre['data_path'] = training_ready_data_path  # replaces raw data path with processed

        # Construct PretrainingConfig for mini-trainer if pretraining mode
        if algorithm_params.get('is_pretraining', False):
            if (block_size := algorithm_params.get('block_size', None)) is None:
                raise ValueError('block_size is required when is_pretraining=True')
            pretraining_config = PretrainingConfig(
                block_size=block_size,
            )
            training_args_pre['pretraining_config'] = pretraining_config

        # mini trainer can support multiple modes, but we don't expose this feature by default
        # to prevent the current API from becoming overly complicated
        if not isinstance(
            train_mode := training_args_pre.get('training_mode', TrainingMode.EPOCH),
            TrainingMode,
        ):
            train_mode = TrainingMode(train_mode)
        training_args_pre['training_mode'] = train_mode

        # user may want to control this API field for debug purposes, so we allow for it to be read
        # but default it to True
        training_args_pre['osft'] = training_args_pre.get('osft', True)

        # now we run training
        return run_training(
            torch_args=torch_args,
            train_args=TrainingArgs(**training_args_pre),
        )

    def _process_data(
        self,
        model_name_or_path: str,
        data_path: str,
        output_dir: str,
        max_seq_len: int,
        num_cpu_procs: int,
        unmask_messages: bool,
        use_processed_dataset: bool,
        is_pretraining: bool = False,
        document_column_name: str | None = None,
    ) -> str:
        """
        Process the data into a format that can be used for training.

        Returns the path to the processed dataset.
        """
        # mini trainer doesn't do its own data processing, so we use the one from
        # instructlab training
        from instructlab.training.data_process import (
            process_messages_into_input_ids,
            process_documents_for_pretraining,
        )

        # if we're using the processed dataset, then we don't need to do any data processing
        if use_processed_dataset:
            return data_path

        # otherwise we need to process the data
        os.makedirs(output_dir, exist_ok=True)

        # Handle pretraining mode
        if is_pretraining:
            # pass any optional kwargs as-needed
            additional_kwargs = {}
            if document_column_name is not None:
                additional_kwargs['document_column_name'] = document_column_name

            process_documents_for_pretraining(
                data_path=data_path,
                data_output_path=output_dir,
                model_path=model_name_or_path,
                num_cpu_procs=num_cpu_procs,
                **additional_kwargs,
            )
        else:
            # Original instruction tuning flow
            # if we received unmask then we need to add that
            processing_data_path = data_path
            if unmask_messages:
                ds = datasets.load_dataset('json', data_files=data_path, split='train')
                ds = ds.map(lambda _: {'unmask': True})
                processing_data_path = os.path.join(output_dir, 'intermediate_data.jsonl')
                ds.to_json(processing_data_path)

            # now we process the data
            process_messages_into_input_ids(
                data_path=processing_data_path,
                data_output_path=output_dir,
                model_path=model_name_or_path,
                max_seq_len=max_seq_len,
                num_cpu_procs=num_cpu_procs,
            )

        # above function will save to this file, so we pass this to the trainer
        return os.path.join(output_dir, 'data.jsonl')


AlgorithmRegistry.register_algorithm('osft', OSFTAlgorithm)
AlgorithmRegistry.register_backend('osft', 'mini-trainer', MiniTrainerOSFTBackend)


def osft(
    model_path: str,
    data_path: str,
    unfreeze_rank_ratio: float,
    effective_batch_size: int,
    max_tokens_per_gpu: int,
    max_seq_len: int,
    learning_rate: float,
    ckpt_output_dir: str,
    data_output_dir: str | None = None,
    backend: str = 'mini-trainer',
    # Optional parameters
    target_patterns: list[str] | None = None,
    seed: int | None = None,
    use_liger: bool | None = None,
    use_processed_dataset: bool | None = None,
    unmask_messages: bool | None = None,
    is_pretraining: bool | None = None,
    block_size: int | None = None,
    document_column_name: str | None = None,
    lr_scheduler: str | None = None,
    warmup_steps: int | None = None,
    lr_scheduler_kwargs: dict[str, str] | None = None,
    # AdamW optimizer parameters
    beta1: float | None = None,
    beta2: float | None = None,
    eps: float | None = None,
    weight_decay: float | None = None,
    checkpoint_at_epoch: bool | None = None,
    save_final_checkpoint: bool | None = None,
    num_epochs: int | None = None,
    # Torchrun parameters for multi-node support
    nproc_per_node: Literal['auto', 'gpu'] | int | None = None,
    nnodes: int | None = None,
    node_rank: int | None = None,
    rdzv_id: str | int | None = None,
    rdzv_endpoint: str | None = None,
    master_port: int | None = None,
    master_addr: str | None = None,
    # Logging parameters
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    tensorboard_log_dir: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    **kwargs,
) -> any:
    from . import create_algorithm

    algorithm: OSFTAlgorithm = create_algorithm('osft', backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        data_output_dir=data_output_dir,
        unfreeze_rank_ratio=unfreeze_rank_ratio,
        effective_batch_size=effective_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        target_patterns=target_patterns,
        seed=seed,
        use_liger=use_liger,
        use_processed_dataset=use_processed_dataset,
        unmask_messages=unmask_messages,
        is_pretraining=is_pretraining,
        block_size=block_size,
        document_column_name=document_column_name,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        checkpoint_at_epoch=checkpoint_at_epoch,
        save_final_checkpoint=save_final_checkpoint,
        num_epochs=num_epochs,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        rdzv_id=rdzv_id,
        rdzv_endpoint=rdzv_endpoint,
        master_port=master_port,
        master_addr=master_addr,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        tensorboard_log_dir=tensorboard_log_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        **kwargs,
    )
