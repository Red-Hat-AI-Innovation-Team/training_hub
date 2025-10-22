#!/usr/bin/env python3
"""
Example usage of the LoRA algorithm in training hub.

This example demonstrates both basic LoRA fine-tuning and QLoRA (4-bit quantization).
"""

import training_hub


def basic_lora_sft_example():
    """Basic LoRA + SFT fine-tuning example using Unsloth backend."""
    print("🚀 Running basic LoRA + SFT fine-tuning example (Unsloth backend)...")

    try:
        result = training_hub.lora_sft(
            model_path="microsoft/DialoGPT-small",  # Small model for quick testing
            data_path="./sample_data.jsonl",  # Your training data
            ckpt_output_dir="./outputs/lora_basic",
            backend="unsloth",  # Default backend with performance optimizations

            # LoRA configuration
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,

            # Training configuration
            num_epochs=1,
            learning_rate=2e-4,
            max_seq_len=512,
            micro_batch_size=2,

            # Optimization (Unsloth optimizations are automatic)
            bf16=True,
            sample_packing=True,

            # Logging
            logging_steps=10,
            save_steps=100,
        )

        print("✅ Basic LoRA + SFT training completed successfully!")
        print(f"Model: {type(result['model'])}")
        print(f"Tokenizer: {type(result['tokenizer'])}")

    except Exception as e:
        print(f"❌ Basic LoRA + SFT training failed: {e}")


def backend_comparison_example():
    """Compare different LoRA SFT backends."""
    print("🚀 Demonstrating different LoRA + SFT backends...")

    # Example 1: Unsloth Backend (default - optimized for performance)
    print("\n1️⃣ Unsloth backend (performance optimized):")
    try:
        result = training_hub.lora_sft(
            model_path="microsoft/DialoGPT-small",
            data_path="./sample_data.jsonl",
            ckpt_output_dir="./outputs/lora_unsloth",
            backend="unsloth",  # Performance optimized with custom kernels

            lora_r=8,
            num_epochs=1,
            micro_batch_size=1,
            learning_rate=2e-4,
            load_in_4bit=True,  # Automatic memory optimization
        )
        print("✅ Unsloth backend training completed!")
    except Exception as e:
        print(f"❌ Unsloth backend training failed: {e}")

    # Example 2: Axolotl Backend (comprehensive features, but may have dependency conflicts)
    print("\n2️⃣ Axolotl backend (comprehensive features):")
    try:
        result = training_hub.lora_sft(
            model_path="microsoft/DialoGPT-small",
            data_path="./sample_data.jsonl",
            ckpt_output_dir="./outputs/lora_axolotl",
            backend="axolotl",  # Full-featured framework

            # Note: May require resolving dependency conflicts
            lora_r=8,
            num_epochs=1,
            micro_batch_size=1,
            learning_rate=2e-4,
        )
        print("✅ Axolotl backend training completed!")
    except Exception as e:
        print(f"❌ Axolotl backend training failed: {e}")
        print("💡 Tip: Axolotl may have dependency conflicts. Try installing separately or use Unsloth backend.")


def qlora_4bit_example():
    """QLoRA + SFT with 4-bit quantization example using Unsloth."""
    print("🚀 Running QLoRA + SFT (4-bit quantization) example with Unsloth optimizations...")

    try:
        result = training_hub.lora_sft(
            model_path="microsoft/DialoGPT-medium",
            data_path="./sample_data.jsonl",
            ckpt_output_dir="./outputs/qlora_4bit",
            backend="unsloth",  # Unsloth provides optimized QLoRA

            # LoRA configuration
            lora_r=64,  # Higher rank for quantized model
            lora_alpha=128,
            lora_dropout=0.1,

            # QLoRA (4-bit quantization) - Unsloth handles optimization automatically
            load_in_4bit=True,

            # Training configuration
            num_epochs=2,
            learning_rate=1e-4,  # Lower learning rate for quantized training
            max_seq_len=1024,
            micro_batch_size=1,  # Smaller batch for memory efficiency

            # Optimization (Unsloth provides automatic optimizations)
            bf16=True,
            sample_packing=True,

            # Logging
            logging_steps=5,
            save_steps=50,
        )

        print("✅ QLoRA 4-bit training completed successfully!")
        print(f"Model: {type(result['model'])}")
        print(f"Tokenizer: {type(result['tokenizer'])}")
        print("🚀 Unsloth provided automatic memory and speed optimizations!")

    except Exception as e:
        print(f"❌ QLoRA 4-bit training failed: {e}")


def distributed_lora_sft_example():
    """Multi-GPU LoRA + SFT training example."""
    print("🚀 Running distributed LoRA + SFT training example...")

    try:
        result = training_hub.lora_sft(
            model_path="meta-llama/Llama-2-7b-hf",
            data_path="./large_dataset.jsonl",
            ckpt_output_dir="./outputs/lora_distributed",

            # LoRA configuration
            lora_r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Specify target modules

            # Training configuration
            num_epochs=3,
            effective_batch_size=128,  # Will be split across GPUs
            learning_rate=2e-4,
            max_seq_len=2048,

            # Distributed training
            nproc_per_node=4,  # 4 GPUs per node
            nnodes=1,  # Single node

            # Optimization
            bf16=True,
            flash_attention=True,
            sample_packing=True,

            # Weights & Biases logging
            wandb_project="lora_training",
            wandb_entity="your_team",

            # Checkpointing
            save_steps=200,
            save_total_limit=3,
        )

        print("✅ Distributed LoRA training completed successfully!")

    except Exception as e:
        print(f"❌ Distributed LoRA training failed: {e}")


def create_sample_data():
    """Create sample datasets in different formats for testing."""
    import json
    import os

    # Messages format (compatible with instructlab-training)
    messages_data = [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain what LoRA is."},
                {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adapts large language models by learning low-rank decomposition matrices."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."},
                {"role": "assistant", "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"}
            ]
        }
    ]

    # Alpaca format
    alpaca_data = [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain what LoRA is.",
            "input": "",
            "output": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adapts large language models by learning low-rank decomposition matrices."
        },
        {
            "instruction": "Write a Python function to calculate fibonacci numbers.",
            "input": "",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        }
    ]

    os.makedirs("./", exist_ok=True)

    # Create messages format file (default for compatibility)
    with open("./sample_data.jsonl", "w") as f:
        for item in messages_data:
            f.write(json.dumps(item) + "\n")

    # Create alpaca format file
    with open("./sample_data_alpaca.jsonl", "w") as f:
        for item in alpaca_data:
            f.write(json.dumps(item) + "\n")

    print("📝 Created sample_data.jsonl (messages format)")
    print("📝 Created sample_data_alpaca.jsonl (alpaca format)")


def data_format_examples():
    """Demonstrate different data format options."""
    print("🚀 Demonstrating different data format support...")

    # Example 1: Messages format (default, compatible with instructlab-training)
    print("\n1️⃣ Messages format (default):")
    try:
        result = training_hub.lora_sft(
            model_path="microsoft/DialoGPT-small",
            data_path="./sample_data.jsonl",  # Messages format
            ckpt_output_dir="./outputs/lora_messages",

            # No dataset_type specified - defaults to 'chat_template' for messages
            lora_r=8,
            num_epochs=1,
            micro_batch_size=1,
            learning_rate=2e-4,
        )
        print("✅ Messages format training completed!")
    except Exception as e:
        print(f"❌ Messages format training failed: {e}")

    # Example 2: Alpaca format
    print("\n2️⃣ Alpaca format:")
    try:
        result = training_hub.lora_sft(
            model_path="microsoft/DialoGPT-small",
            data_path="./sample_data_alpaca.jsonl",  # Alpaca format
            ckpt_output_dir="./outputs/lora_alpaca",

            # Specify alpaca dataset type
            dataset_type="alpaca",
            lora_r=8,
            num_epochs=1,
            micro_batch_size=1,
            learning_rate=2e-4,
        )
        print("✅ Alpaca format training completed!")
    except Exception as e:
        print(f"❌ Alpaca format training failed: {e}")

    # Example 3: Custom field mapping
    print("\n3️⃣ Custom field mapping:")
    try:
        result = training_hub.lora_sft(
            model_path="microsoft/DialoGPT-small",
            data_path="./sample_data_alpaca.jsonl",  # Using alpaca data with custom mapping
            ckpt_output_dir="./outputs/lora_custom",

            # Custom field mapping
            dataset_type="alpaca",
            field_instruction="instruction",
            field_input="input",
            field_output="output",

            lora_r=8,
            num_epochs=1,
            micro_batch_size=1,
            learning_rate=2e-4,
        )
        print("✅ Custom field mapping training completed!")
    except Exception as e:
        print(f"❌ Custom field mapping training failed: {e}")


if __name__ == "__main__":
    print("🎯 Training Hub LoRA Examples")
    print("=" * 50)

    # Create sample data
    create_sample_data()

    # Run examples (uncomment the ones you want to test)
    basic_lora_sft_example()

    # Uncomment to test different backends
    # backend_comparison_example()

    # Uncomment to test different data formats
    # data_format_examples()

    # qlora_4bit_example()
    # distributed_lora_sft_example()

    print("\n✨ Examples completed!")
    print("\nTo run these examples:")
    print("1. Install training-hub with: pip install -e .")
    print("2. For CUDA optimizations: pip install -e .[cuda]")
    print("3. Install TRL for Unsloth backend: pip install trl")
    print("4. Prepare your training data in messages or Alpaca format")
    print("5. Adjust model paths and data paths as needed")
    print("6. Run: python examples/lora_example.py")
    print("\n🚀 Benefits:")
    print("• Unsloth backend: 2x faster, 70% less VRAM, works with any HuggingFace model")
    print("• Supports both LoRA and QLoRA with automatic optimizations")
    print("• Compatible with existing SFT/OSFT data formats")