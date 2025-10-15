from typing import Any, Dict, Type, Optional
from instructlab.training import run_training, TorchrunArgs, TrainingArgs

from . import Algorithm, Backend, AlgorithmRegistry
from training_hub import utils


class InstructLabTrainingSFTBackend(Backend):
    """InstructLab Training backend for SFT algorithm."""
    
    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute SFT training using instructlab-training."""
        # Separate torchrun parameters from training parameters
        torchrun_keys = {'nproc_per_node', 'nnodes', 'node_rank', 'rdzv_id', 'rdzv_endpoint', 'master_addr', 'master_port'}
        
        # Extract torchrun parameters
        torchrun_params = {k: v for k, v in algorithm_params.items() if k in torchrun_keys}
        
        # Extract training parameters (everything except torchrun params)
        training_params = {k: v for k, v in algorithm_params.items() if k not in torchrun_keys}
        
        # Map training_hub parameter names to instructlab-training parameter names
        if 'max_tokens_per_gpu' in training_params:
            training_params['max_batch_len'] = training_params.pop('max_tokens_per_gpu')
        
        # Create TrainingArgs with all provided parameters, letting it handle defaults
        training_args = TrainingArgs(**training_params)
        
        # Set up torchrun arguments with single-node defaults (except nproc_per_node)
        final_torchrun_params = utils.get_torchrun_params(torchrun_params)
        torchrun_args = TorchrunArgs(**final_torchrun_params)

        # Execute training
        return run_training(
            torch_args=torchrun_args,
            train_args=training_args
        )


class SFTAlgorithm(Algorithm):
    """Supervised Fine-Tuning algorithm."""
    
    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs
    
    def train(self, 
              model_path: str,
              data_path: str, 
              ckpt_output_dir: str,
              # Training parameters (defaults from TrainingArgs)
              num_epochs: Optional[int] = None,
              effective_batch_size: Optional[int] = None,
              learning_rate: Optional[float] = None,
              max_seq_len: Optional[int] = None,
              max_tokens_per_gpu: Optional[int] = None,
              data_output_dir: Optional[str] = None,
              save_samples: Optional[int] = None,
              warmup_steps: Optional[int] = None,
              accelerate_full_state_at_epoch: Optional[bool] = None,
              checkpoint_at_epoch: Optional[bool] = None,
              # Torchrun parameters for multi-node support
              nproc_per_node: Optional[str | int] = None,
              nnodes: Optional[int] = None,
              node_rank: Optional[int] = None,
              rdzv_id: Optional[str | int] = None,
              rdzv_endpoint: Optional[str] = None,
              master_addr: Optional[str] = None,
              master_port: Optional[int] = None,
              **kwargs) -> Any:
        """Execute SFT training.
        
        Args:
            model_path: Path to the model to fine-tune
            data_path: Path to the training data
            ckpt_output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            effective_batch_size: Effective batch size for training
            learning_rate: Learning rate for training
            max_seq_len: Maximum sequence length
            max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.
            data_output_dir: Directory to save processed data
            save_samples: Number of samples to save after training (0 disables saving based on sample count)
            warmup_steps: Number of warmup steps
            accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption
            checkpoint_at_epoch: Whether to checkpoint at each epoch
            nproc_per_node: Number of processes (GPUs) per node
            nnodes: Total number of nodes
            node_rank: Rank of this node (0 to nnodes-1)
            rdzv_id: Unique job ID for rendezvous
            rdzv_endpoint: Master node endpoint for multi-node training
            master_addr: Master node address for distributed training
            master_port: Master node port for distributed training
            **kwargs: Additional parameters passed to the backend
            
        Returns:
            Training result from the backend
        """
        # Build parameters dict, only including non-None values
        params = {'model_path': model_path, 'data_path': data_path, 'ckpt_output_dir': ckpt_output_dir}
        
        # Add optional parameters if provided
        optional_params = {
            'num_epochs': num_epochs,
            'effective_batch_size': effective_batch_size,
            'learning_rate': learning_rate,
            'max_seq_len': max_seq_len,
            'max_tokens_per_gpu': max_tokens_per_gpu,
            'data_output_dir': data_output_dir,
            'save_samples': save_samples,
            'warmup_steps': warmup_steps,
            'accelerate_full_state_at_epoch': accelerate_full_state_at_epoch,
            'checkpoint_at_epoch': checkpoint_at_epoch,
            'nproc_per_node': nproc_per_node,
            'nnodes': nnodes,
            'node_rank': node_rank,
            'rdzv_id': rdzv_id,
            'rdzv_endpoint': rdzv_endpoint,
            'master_addr': master_addr,
            'master_port': master_port,
        }
        
        # Only add non-None parameters (let TrainingArgs handle defaults)
        for key, value in optional_params.items():
            if value is not None:
                params[key] = value
                
        params.update(kwargs)
        
        return self.backend.execute_training(params)
    
    def get_required_params(self) -> Dict[str, Type]:
        """Return required parameters for SFT."""
        return {
            'model_path': str,
            'data_path': str,
            'ckpt_output_dir': str,
            'num_epochs': int,
            'effective_batch_size': int,
            'learning_rate': float,
            'max_seq_len': int,
            'max_batch_len': int,
        }

    def get_optional_params(self) -> Dict[str, Type]:
        """Return optional parameters for SFT."""
        return {
            'max_tokens_per_gpu': int,
            'data_output_dir': str,
            'save_samples': int,
            'warmup_steps': int,
            'accelerate_full_state_at_epoch': bool,
            'checkpoint_at_epoch': bool,
            'nproc_per_node': str | int,
            'nnodes': int,
            'node_rank': int,
            'rdzv_id': str | int,
            'rdzv_endpoint': str,
            'master_addr': str,
            'master_port': int,
        }


# Register the algorithm and backend
AlgorithmRegistry.register_algorithm('sft', SFTAlgorithm)
AlgorithmRegistry.register_backend('sft', 'instructlab-training', InstructLabTrainingSFTBackend)


def sft(model_path: str, 
        data_path: str, 
        ckpt_output_dir: str,
        backend: str = "instructlab-training",
        # Training parameters (defaults from TrainingArgs)
        num_epochs: Optional[int] = None,
        effective_batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        max_tokens_per_gpu: Optional[int] = None,
        data_output_dir: Optional[str] = None,
        save_samples: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        accelerate_full_state_at_epoch: Optional[bool] = None,
        checkpoint_at_epoch: Optional[bool] = None,
        # Torchrun parameters for multi-node support
        nproc_per_node: Optional[str | int] = None,
        nnodes: Optional[int] = None,
        node_rank: Optional[int] = None,
        rdzv_id: Optional[str | int] = None,
        rdzv_endpoint: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **kwargs) -> Any:
    """Convenience function to run SFT training.
    
    Args:
        model_path: Path to the model to fine-tune
        data_path: Path to the training data
        ckpt_output_dir: Directory to save checkpoints
        backend: Backend implementation to use (default: "instructlab-training")
        num_epochs: Number of training epochs
        effective_batch_size: Effective batch size for training
        learning_rate: Learning rate for training
        max_seq_len: Maximum sequence length
        max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.
        data_output_dir: Directory to save processed data
        save_samples: Number of samples to save after training (0 disables saving based on sample count)
        warmup_steps: Number of warmup steps
        accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption
        checkpoint_at_epoch: Whether to checkpoint at each epoch
        nproc_per_node: Number of processes (GPUs) per node for distributed training
        nnodes: Total number of nodes for distributed training
        node_rank: Rank of this node (0 to nnodes-1) for distributed training
        rdzv_id: Unique job ID for rendezvous in distributed training
        rdzv_endpoint: Master node endpoint for multi-node training
        master_addr: Master node address for distributed training
        master_port: Master node port for distributed training

        **kwargs: Additional parameters passed to the backend
    
    Returns:
        Training result from the backend
    """
    from . import create_algorithm
    
    algorithm = create_algorithm('sft', backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        num_epochs=num_epochs,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=max_tokens_per_gpu,
        data_output_dir=data_output_dir,
        save_samples=save_samples,
        warmup_steps=warmup_steps,
        accelerate_full_state_at_epoch=accelerate_full_state_at_epoch,
        checkpoint_at_epoch=checkpoint_at_epoch,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        rdzv_id=rdzv_id,
        rdzv_endpoint=rdzv_endpoint,
        master_addr=master_addr,
        master_port=master_port,
        **kwargs
    )

