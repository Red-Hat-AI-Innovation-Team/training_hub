#!/usr/bin/env python3
sample_name="SFT Training Example for HPU"
"""
This script demonstrates how to do SFT training on HPU
using a single-node, multi-GPU setup with training_hub.

Example usage:
    python sft_hpu_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
from datetime import datetime
import argparse

from training_hub import sft


def main():
    #disable HPU backend autoloading
    os.environ['PT_HPU_AUTOLOAD'] = '0'

    parser = argparse.ArgumentParser(description=sample_name)
    
    # Required parameters
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (JSONL format)')
    parser.add_argument('--ckpt-output-dir', required=True,
                       help='Directory to save checkpoints')
    parser.add_argument('--model-path', required=True,
                       help='Model path or HuggingFace name')
    
    # Optional overrides
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--effective-batch-size', type=int, default=128,
                       help='effective batch size')
    parser.add_argument('--max-seq-len', type=int, default=16384,
                       help='Maximum sequence length')
    parser.add_argument('--checkpoint-at-epoch', action='store_true',
                       help='Store checkpoint after each epoch')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=32768,
                       help='Max tokens per GPU')
    parser.add_argument('--nproc-per-node', type=int, default=8,
                       help='Number of GPUs')
    parser.add_argument('--torch-compile', action='store_true', default=False,
                       help='Enable torch.compile, hpu only')    
    parser.add_argument('--num-chunks', type=int, default=1,
                       help='Number of chunks to split dataset into for sequential training')
    
    args = parser.parse_args()

    # sample configuration
    print(f"🚀 {sample_name}")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print()
    
    # Training configuration optimized for Llama 3.1 8B Instruct
    start_time = time.time()
    
    try:
        result = sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            
            # Training parameters
            num_epochs=args.num_epochs,
            effective_batch_size=args.effective_batch_size,
            learning_rate=1e-5,                # Lower LR for instruct model
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data processing
            data_output_dir="/dev/shm",        # Use RAM disk for speed
            warmup_steps=100,
            save_samples=0,                    # 0 disables sample-based checkpointing, use epoch-based only
            
            # Checkpointing
            checkpoint_at_epoch=args.checkpoint_at_epoch,
            accelerate_full_state_at_epoch=False, # Disable for smaller checkpoints (no auto-resumption)
            
            # Single-node multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=101,
            rdzv_endpoint="127.0.0.1:29500",

            # HPU specific arguments
            disable_flash_attn = True,
            device = 'hpu',
            torch_compile = args.torch_compile,
            num_chunks = args.num_chunks,
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 50)
        print("✅ Training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format/")
        
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