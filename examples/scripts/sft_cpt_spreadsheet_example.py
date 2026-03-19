#!/usr/bin/env python3
"""
Continued Pretraining Example: Excel Spreadsheets with Granite 3.3 8B Instruct

This script demonstrates how to use Training Hub's pretraining mode to do
continued pretraining (CPT) on an instruct model using parsed Excel spreadsheet
data. It downloads the SpreadsheetBench dataset (real-world .xlsx files),
converts them to markdown text using Microsoft's markitdown library, and
trains using SFT in pretraining mode.

The pipeline:
    1. Download SpreadsheetBench dataset from Hugging Face
    2. Extract .xlsx files from the archive
    3. Convert each spreadsheet to markdown text via markitdown
    4. Write a JSONL file with {"document": "..."} entries
    5. Run SFT with is_pretraining=True for continued pretraining

Prerequisites:
    pip install 'markitdown[xlsx]' huggingface_hub

Example usage:
    # Prepare data and train in one step
    python sft_cpt_spreadsheet_example.py \\
        --ckpt-output-dir /path/to/checkpoints

    # Use pre-prepared data (skip download/conversion)
    python sft_cpt_spreadsheet_example.py \\
        --data-path /path/to/spreadsheets.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # Prepare data only (no training)
    python sft_cpt_spreadsheet_example.py --prepare-only
"""

import os
import sys
import json
import time
import glob
import tarfile
import argparse
from datetime import datetime
from pathlib import Path

import torch

from training_hub import sft


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

DEFAULT_MODEL_PATH = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BLOCK_SIZE = 2048
DEFAULT_NPROC_PER_NODE = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1

# Pretraining hyperparameters — lower learning rate than instruction tuning
# since we are injecting new knowledge into an already-trained model
DEFAULT_LEARNING_RATE = 2e-6
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_TOKENS_PER_GPU = 25000
DEFAULT_MAX_SEQ_LEN = 4096


# =============================================================================
# DATA PREPARATION
# =============================================================================

DATASET_REPO = "KAKA22/SpreadsheetBench"
DATASET_FILE = "spreadsheetbench_verified_400.tar.gz"


def download_dataset(cache_dir: str) -> str:
    """Download SpreadsheetBench from Hugging Face.

    Args:
        cache_dir: Directory for caching downloaded files.

    Returns:
        Path to the downloaded tar.gz file.
    """
    from huggingface_hub import hf_hub_download

    print(f"Downloading {DATASET_FILE} from {DATASET_REPO}...")
    filepath = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded to: {filepath}")
    return filepath


def extract_dataset(tar_path: str, extract_dir: str) -> str:
    """Extract the tar.gz archive.

    Args:
        tar_path: Path to the downloaded tar.gz file.
        extract_dir: Directory to extract into.

    Returns:
        Path to the extraction root directory.
    """
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        base_dir = Path(extract_dir).resolve()
        safe_members: list[tarfile.TarInfo] = []
        for member in tar.getmembers():
            target_path = (base_dir / member.name).resolve()
            if os.path.commonpath([str(base_dir), str(target_path)]) != str(base_dir):
                raise ValueError(f"Unsafe tar member path: {member.name}")
            safe_members.append(member)
        tar.extractall(path=extract_dir, members=safe_members)
    print(f"  Extracted to: {extract_dir}")
    return extract_dir


def find_xlsx_files(extract_dir: str) -> list[str]:
    """Find all .xlsx files in the extracted dataset.

    Args:
        extract_dir: Root directory of the extracted dataset.

    Returns:
        List of paths to .xlsx files.
    """
    xlsx_files = sorted(glob.glob(
        os.path.join(extract_dir, "**", "*.xlsx"),
        recursive=True,
    ))
    # SpreadsheetBench has _init.xlsx (input) and _golden.xlsx (expected output).
    # Use only _init.xlsx files to avoid training on answer sheets.
    init_files = [f for f in xlsx_files if "_init.xlsx" in f]
    # Fall back to all xlsx files if naming convention doesn't match
    if not init_files:
        input_files = xlsx_files
    else:
        input_files = init_files
    print(f"  Found {len(input_files)} spreadsheet files")
    return input_files


def convert_xlsx_to_text(xlsx_path: str) -> str | None:
    """Convert an Excel spreadsheet to markdown text.

    Uses Microsoft's markitdown library to produce LLM-readable
    markdown tables from spreadsheet data.

    Args:
        xlsx_path: Path to the .xlsx file.

    Returns:
        Markdown text content, or None if conversion fails.
    """
    from markitdown import MarkItDown

    md = MarkItDown(enable_plugins=False)
    try:
        result = md.convert(xlsx_path)
        text = result.text_content.strip()
        return text if text else None
    except Exception as e:
        print(f"  Warning: skipping {os.path.basename(xlsx_path)}: {e}")
        return None


def prepare_data(
    output_jsonl: str,
    cache_dir: str = "./spreadsheet_cache",
) -> str:
    """Download, extract, convert, and write the training JSONL.

    Args:
        output_jsonl: Path for the output JSONL file.
        cache_dir: Directory for caching intermediate files.

    Returns:
        Path to the written JSONL file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    extract_dir = os.path.join(cache_dir, "extracted")

    # Download
    tar_path = download_dataset(cache_dir)

    # Extract
    extract_dataset(tar_path, extract_dir)

    # Find xlsx files
    xlsx_files = find_xlsx_files(extract_dir)
    if not xlsx_files:
        print("Error: no .xlsx files found in the dataset")
        sys.exit(1)

    # Convert and write JSONL
    print(f"Converting {len(xlsx_files)} spreadsheets to text...")
    doc_count = 0
    skip_count = 0

    with open(output_jsonl, "w") as out:
        for i, xlsx_path in enumerate(xlsx_files, 1):
            text = convert_xlsx_to_text(xlsx_path)
            if text:
                record = {"document": text}
                out.write(json.dumps(record) + "\n")
                doc_count += 1
            else:
                skip_count += 1

            if i % 50 == 0:
                print(f"  Processed {i}/{len(xlsx_files)} files...")

    print("Data preparation complete:")
    print(f"  Documents written: {doc_count}")
    print(f"  Files skipped: {skip_count}")
    print(f"  Output: {output_jsonl}")

    if doc_count == 0:
        print("Error: no documents were successfully converted")
        sys.exit(1)

    return output_jsonl


# =============================================================================
# TRAINING
# =============================================================================

def main() -> None:
    """Run the spreadsheet continued pretraining CLI workflow."""
    parser = argparse.ArgumentParser(
        description="Continued Pretraining on Excel Spreadsheets with Training Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        help="Path to pre-prepared JSONL file. If not provided, data will be "
             "downloaded and prepared automatically.",
    )
    parser.add_argument(
        "--ckpt-output-dir",
        help="Directory to save training checkpoints (required unless --prepare-only)",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Only prepare the data (download, convert), skip training",
    )
    parser.add_argument(
        "--output-jsonl", default="./spreadsheet_pretraining_data.jsonl",
        help="Output path for the prepared JSONL file (default: ./spreadsheet_pretraining_data.jsonl)",
    )
    parser.add_argument(
        "--cache-dir", default="./spreadsheet_cache",
        help="Cache directory for downloaded/extracted files (default: ./spreadsheet_cache)",
    )

    # Model arguments
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH,
        help=f"Model path or HuggingFace name (default: {DEFAULT_MODEL_PATH})",
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for pretraining document packing in tokens (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE,
        help=f"Number of GPUs (default: {DEFAULT_NPROC_PER_NODE})",
    )
    parser.add_argument(
        "--max-tokens-per-gpu", type=int, default=DEFAULT_MAX_TOKENS_PER_GPU,
        help=f"Max tokens per GPU per step (default: {DEFAULT_MAX_TOKENS_PER_GPU})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Effective batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LEN})",
    )

    args = parser.parse_args()

    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")

    # -------------------------------------------------------------------------
    # Step 1: Prepare data if needed
    # -------------------------------------------------------------------------
    if args.data_path and os.path.exists(args.data_path):
        data_path = args.data_path
        print(f"Using existing data file: {data_path}")
    else:
        if args.data_path:
            print(f"Data file not found at {args.data_path}, preparing from scratch...")
        data_path = prepare_data(
            output_jsonl=args.output_jsonl,
            cache_dir=args.cache_dir,
        )

    if args.prepare_only:
        print("--prepare-only set, skipping training.")
        return

    # -------------------------------------------------------------------------
    # Step 2: Train
    # -------------------------------------------------------------------------
    if not args.ckpt_output_dir:
        parser.error("--ckpt-output-dir is required for training")

    experiment_name = "cpt_spreadsheet"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_output_dir = f"data/{experiment_name}_{timestamp}"

    print()
    print("=" * 60)
    print("Continued Pretraining: Excel Spreadsheets")
    print("=" * 60)
    print(f"  Model:              {args.model_path}")
    print(f"  Data:               {data_path}")
    print(f"  Output:             {args.ckpt_output_dir}")
    print(f"  GPUs:               {args.nproc_per_node}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Block size:         {args.block_size}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Learning rate:      {args.learning_rate}")
    print(f"  Max seq len:        {args.max_seq_len:,}")
    print(f"  Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print()

    start_time = time.time()

    try:
        sft(
            # Model and data
            model_path=args.model_path,
            data_path=data_path,
            ckpt_output_dir=args.ckpt_output_dir,

            # Pretraining mode
            is_pretraining=True,
            block_size=args.block_size,
            document_column_name="document",

            # Training parameters — conservative for CPT on instruct models
            num_epochs=args.num_epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,

            # Data processing
            data_output_dir=data_output_dir,
            warmup_steps=50,
            save_samples=0,

            # Checkpointing
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=False,

            # Distributed setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
        )

        end_time = time.time()
        duration = end_time - start_time

        print("=" * 60)
        print("Training completed successfully!")
        print(f"  Duration: {duration/3600:.2f} hours")
        print(f"  Checkpoints: {args.ckpt_output_dir}/hf_format/")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print("=" * 60)
        print(f"Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - OOM? Try reducing --max-tokens-per-gpu or --block-size")
        print("  - Too few GPUs? Try increasing --nproc-per-node")
        print("  - Data issues? Run with --prepare-only first to inspect the JSONL")
        sys.exit(1)


if __name__ == "__main__":
    main()
