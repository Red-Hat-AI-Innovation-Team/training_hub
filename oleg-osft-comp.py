#!/usr/bin/env python3
"""
Comparison script using OSFT/mini_trainer with same simple dataset as HF Trainer.
This runs in pure SFT mode (osft=False) for direct comparison.
"""
import os
import json
import argparse
from pathlib import Path
import transformers
from training_hub.algorithms.osft import osft


def create_simple_dataset(tokenizer, block_size, output_path):
    """
    Create a simple pre-tokenized dataset for overfitting test.
    Saves the dataset as a jsonl file with input_ids and labels.
    """
    # Create a single canonical text repeated to remove any variation across batches
    # canonical_text = "The quick brown fox jumps over the lazy dog. " * 10
    # simple_texts = [canonical_text] * 15  # 15 identical examples
    canonical_text = "ab"
    simple_texts = [canonical_text] * 15  # 15 identical examples

    # Pre-tokenize everything, leaving room for EOS token
    tokenized = tokenizer(simple_texts, truncation=True, max_length=block_size - 1, padding=False)

    # Add EOS token to the end of each sequence and pad
    samples = []
    for ids in tokenized["input_ids"]:
        # Add EOS token
        ids_with_eos = ids + [tokenizer.eos_token_id]

        # Pad to block_size
        padding_length = block_size - len(ids_with_eos)
        padded_ids = ids_with_eos + [tokenizer.pad_token_id] * padding_length
        label_ids = ids_with_eos + [-100] * padding_length
        attention_mask = [1] * len(ids_with_eos) + [0] * padding_length

        # mini_trainer expects input_ids and labels
        samples.append({
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,  # For causal LM, labels are the same as input_ids
        })

    # Save to jsonl
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Created dataset with {len(samples)} samples at {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 with OSFT in SFT-only mode")

    # Model and data params
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--block_size", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store processed data")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--effective_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_tokens_per_gpu", type=int, default=512, help="Max tokens per GPU (per microbatch)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument("--output_dir", type=str, default="./test-output-osft", help="Output directory")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="cpt-testing", help="Wandb project name")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    # Set pad token if it doesn't exist (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create processed dataset
    data_path = os.path.join(args.data_dir, "simple_dataset.jsonl")
    create_simple_dataset(tokenizer, args.block_size, data_path)

    # Use padded batches (no packing) to mirror HF Trainer attention patterns
    # os.environ["MINITRAINER_PADDED_BATCHES"] = "1"
    # os.environ["MINITRAINER_HF_PARITY"] = "1"

    # Calculate effective batch size - using 1 GPU only
    effective_batch_size = args.effective_batch_size
    # Avoid packed sequences exceeding block_size (matches HF padded batches for GPT-2)
    safe_max_tokens_per_gpu = args.max_tokens_per_gpu

    print(f"\nStarting OSFT training in SFT-only mode (1 GPU):")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {data_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {effective_batch_size}")
    print(f"  Max tokens per GPU: {safe_max_tokens_per_gpu}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Wandb project: {args.wandb_project}")
    print()

    # Run OSFT in SFT-only mode
    osft(
        model_path=args.model_name,
        data_path=data_path,
        unfreeze_rank_ratio=0.0,  # 0.0 means pure SFT (no OSFT projection)
        effective_batch_size=effective_batch_size,
        max_tokens_per_gpu=safe_max_tokens_per_gpu,
        max_seq_len=args.block_size,
        learning_rate=args.learning_rate,
        ckpt_output_dir=args.output_dir,
        use_processed_dataset=True,  # We pre-tokenized the data
        num_epochs=args.num_train_epochs,
        lr_scheduler="cosine",  # Match HF Trainer
        warmup_steps=0,  # Match HF Trainer
        seed=args.seed,
        wandb_project=args.wandb_project,
        osft=False,  # Disable OSFT, run pure SFT
        save_final_checkpoint=True,
        nproc_per_node=2,  # Use only 1 GPU
    )

    print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
