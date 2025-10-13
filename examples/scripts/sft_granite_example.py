#!/usr/bin/env python3
"""
SFT Training Example: Granite 3.3 8B Instruct

This script demonstrates SFT training with Granite 3.3 8B Instruct model
using a single-node, multi-GPU setup with training_hub.

After the training, the script also creates a merged model with linear interpolation.

Example usage:
    python sft_granite_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
import glob
from datetime import datetime
import argparse
import torch

from training_hub import sft


# =============================================================================
# MODEL CONFIGURATION EXAMPLE
# =============================================================================

# Derived from generic_7b_example in examples/notebooks/sft_comprehensive_tutorial.ipynb
granite_example = {
    "model_name": "Granite 3.3 8B Instruct",
    "model_path": "ibm-granite/granite-3.3-8b-instruct",  # HuggingFace model name or local path
    "example_max_tokens_per_gpu": 25000,
    "example_max_seq_len": 20000,
    "example_batch_size": 256,
    "example_learning_rate": 2e-5,
    "notes": "Good baseline for most 7B instruction-tuned models",
}

selected_example = granite_example  # Change this to your preferred example

model_name = selected_example['model_name']
default_model_path = selected_example['model_path']
default_max_tokens_per_gpu = selected_example['example_max_tokens_per_gpu']
default_max_seq_len = selected_example['example_max_seq_len']
default_batch_size = selected_example['example_batch_size']
default_learning_rate = selected_example['example_learning_rate']
default_num_epochs = 3
default_nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0
default_model_weight = 0.5

# =============================================================================
# COMPLETE SFT PARAMETER CONFIGURATION
# =============================================================================

# Experiment identification
experiment_name = "sft_granite_example"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
full_experiment_name = f"{experiment_name}_{timestamp}"

# data_output_dir=f"data/{full_experiment_name}"  # Directory for processed data
data_output_dir=f"/dev/shm/data/{full_experiment_name}"  # Directory for processed data (RAM disk for speed)


# Copied from examples/scripts/osft_continual_learning_example.py
def find_most_recent_checkpoint(output_dir):
    """
    Find the most recent checkpoint in the training output directory.
    
    Args:
        output_dir (str): Training output directory containing hf_format/ subdirectory
        
    Returns:
        str: Path to the most recent checkpoint
        
    Raises:
        ValueError: If no checkpoints are found
    """
    # Get all checkpoint directories under hf_format
    checkpoint_pattern = os.path.join(output_dir, "hf_format", "samples_*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {os.path.join(output_dir, 'hf_format')}")
    
    # Find the most recently created checkpoint
    most_recent_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
    
    return most_recent_checkpoint


def main():
    parser = argparse.ArgumentParser(description=f'SFT Training Example: {model_name}')
    
    # Required parameters
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (JSONL format)')
    parser.add_argument('--ckpt-output-dir', required=True,
                       help='Directory to save checkpoints')
    
    # Optional overrides
    parser.add_argument('--model-path', default=default_model_path,
                       help=f'Model path or HuggingFace name (default: {default_model_path})')
    parser.add_argument('--num-epochs', type=int, default=default_num_epochs,
                       help=f'Number of training epochs (default: {default_num_epochs})')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=default_max_tokens_per_gpu,
                       help=f'Max tokens per GPU (default: {default_max_tokens_per_gpu})')
    parser.add_argument('--nproc-per-node', type=int, default=default_nproc_per_node,
                       help=f'Number of GPUs (default: {default_nproc_per_node})')
    parser.add_argument('--batch-size', type=int, default=default_batch_size,
                       help=f'Effective batch size for training (default: {default_batch_size})')
    parser.add_argument('--learning-rate', type=float, default=default_learning_rate,
                       help=f'Learning rate for training (default: {default_learning_rate})')
    parser.add_argument('--max-seq-len', type=int, default=default_max_seq_len,
                       help=f'Max sequence length (default: {default_max_seq_len})')
    parser.add_argument('--model-weight', type=float, default=default_model_weight,
                       help=f'Weight for trained model for interpolation (0.0-1.0, default: {default_model_weight})')
    
    args = parser.parse_args()

    if args.nproc_per_node < 4:
        raise ValueError("NPROC_PER_NODE must be larger than or equal to 4")
    
    # Granite 3.3 8B Instruct configuration
    print(f"🚀 SFT Training: {model_name}")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_seq_len:,}")
    print(f"Model weight: {args.model_weight}")
    print()
    
    # Training configuration optimized for Granite 3.3 8B Instruct
    start_time = time.time()
    
    try:
        result = sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            
            # Training parameters optimized for Granite 3.3 8B Instruct
            num_epochs=args.num_epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data processing
            data_output_dir=data_output_dir,
            warmup_steps=100,
            save_samples=0,                    # 0 disables sample-based checkpointing, use epoch-based only
            
            # Checkpointing
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=True,  # Enable for auto-resumption (larger checkpoints)
            
            # Single-node multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=102,
            rdzv_endpoint="127.0.0.1:29500",
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 50)
        print("✅ Training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format/")

        most_recent_checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)
        print(f"   Most recent checkpoint: {most_recent_checkpoint}")

        trained_model_weight = args.model_weight
        if 0.0 < trained_model_weight and trained_model_weight < 1.0:
            from interpolator import interpolate_models

            interp_model_path = interpolate_models(args.model_path, most_recent_checkpoint, trained_model_weight=trained_model_weight)

            print("=" * 50)
            print("✅ Interpolation completed successfully!")
            print(f"   Interpolated model checkpoint: {interp_model_path}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 50)
        print(f"❌ Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("💡 Try reducing --max-tokens-per-gpu if you see OOM errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
