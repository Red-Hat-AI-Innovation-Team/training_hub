# let's overfit to HF trainer
from dataclasses import dataclass, field, asdict
from typing import Optional
import transformers
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import torch.distributed as dist
from datasets import Dataset


def get_simple_data_module(tokenizer, block_size):
    """Create a simple pre-tokenized dataset for overfitting test."""
    # Create a single canonical text repeated to remove any variation across batches
    # canonical_text = "The quick brown fox jumps over the lazy dog. " * 10
    # simple_texts = [canonical_text] * 15  # 15 identical examples
    canonical_text = "ab"
    simple_texts = [canonical_text] * 15  # 15 identical examples


    # Pre-tokenize everything, leaving room for EOS token
    tokenized = tokenizer(simple_texts, truncation=True, max_length=block_size - 1, padding=False)

    # Add EOS token to the end of each sequence and pad
    input_ids_list = []
    label_ids_list = []
    attention_mask_list = []

    for ids in tokenized["input_ids"]:
        # Add EOS token
        ids_with_eos = ids + [tokenizer.eos_token_id]

        # Pad to block_size
        padding_length = block_size - len(ids_with_eos)
        padded_ids = ids_with_eos + [tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * len(ids_with_eos) + [0] * padding_length

        # we dont want to generate the pad token
        label_ids = ids_with_eos + [-100] * padding_length

        # append both of them to the list
        input_ids_list.append(padded_ids)
        label_ids_list.append(label_ids)

        attention_mask_list.append(attention_mask)

    # Create dataset directly from input_ids
    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": label_ids_list,  # For causal LM, labels are the same as input_ids
    })

    return {
        "train_dataset": dataset,
        "eval_dataset": dataset,
    }


@dataclass
class TrainingConfig:
    task_name: str
    split: str
    block_size: int
    rehersal_rate: float
    model_name: str
    subsample_ratio: float
    wandb_project: Optional[str] = field(default="cpt-testing")

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_DISABLED'] = 'false'

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    transformers.set_seed(args.seed)

    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)


    # Set pad token if it doesn't exist (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # loading dataset
    data_module = get_simple_data_module(tokenizer, config.block_size)

    print(f"data_module: {data_module['train_dataset'][0]}")

    # setting up trainer
    trainer = transformers.Trainer(model=model, args=args, **data_module)
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    trainer.accelerator.wait_for_everyone()
    tokenizer.save_pretrained(args.output_dir)
    # Clean up at the end of your script
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
