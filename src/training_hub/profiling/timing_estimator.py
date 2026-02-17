try: from typing import override
except ImportError: from typing_extensions import override
from typing import Any
import warnings
import torch
from transformers import AutoModel, AutoTokenizer
from mini_trainer.osft_utils import MODEL_CONFIGS

import os
import pandas as pd
import math
import sys
import numpy as np
import time
from datetime import timedelta

import json
from datetime import datetime
from pathlib import Path

from training_hub import sft, osft, lora_sft

import glob

from transformers.models.llama.modeling_llama import LlamaModel

"""
Code assisted by Cursor/Claude4
This code is based on the scripts in the examples/scripts folder. 
"""

# TODO: Make sure to adjust the scale correction by basing it on the remaining TOKENS in the dataset
# (this calculation should only take a few seconds...maybe this too can be cached?)

class OnlineTimingEstimator:
    def __init__(self, model_path: str,
                num_gpus: int=2,
                dataset_dir: str="table_ds.jsonl",
                max_tokens_per_gpu: int=10000,
                max_seq_len: int=8192,
                batch_size: int=128,
                lr: float=1e-5,
                unfreeze_rank_ratio: float=0.2,
                ckpt_dir: str = 'timing_temp_ckpts',
                use_processed_dataset=False,
                unmask_messages=False
                ):

        # Let's train on 5% of the data for now...
        # Create a temp jsonl that only has a certain portion of the data
        data = pd.read_json(path_or_buf=dataset_dir, lines=True)
        self.sample_frac = 0.05
        self.remaining_frac = 1.0 - self.sample_frac
        sampled_data = data.sample(frac=self.sample_frac, axis=0)
        self.num_samples = len(sampled_data)
        sampled_data.to_json(path_or_buf='temp_data.jsonl', lines=True, orient='records')

        try: os.mkdir(ckpt_dir)
        except FileExistsError: pass

        self.training_kwargs = {
            # Model and data
            'model_path': model_path,
            'data_path': 'temp_data.jsonl',
            'ckpt_output_dir': ckpt_dir,
                    
            # Training parameters
            'num_epochs': 1,
            'effective_batch_size': batch_size,           # Good balance for instruct models
            'learning_rate': lr,                 # Lower LR for instruct model
            'max_seq_len': max_seq_len,       # Use provided max sequence length
            'max_tokens_per_gpu': max_tokens_per_gpu,
                    
            # Data processing
            'data_output_dir': "/dev/shm",         # Use RAM disk for speed
            'warmup_steps': 100,
            'save_samples': 0,                     # 0 disables sample-based checkpointing
                    
            # Checkpointing
            'checkpoint_at_epoch': False,
            'accelerate_full_state_at_epoch': False,  # Enable for auto-resumption
                    
            # Single-node setup
            'nproc_per_node': num_gpus,
            'nnodes': 1,
            
            # OSFT-specific parameters
            'unfreeze_rank_ratio': unfreeze_rank_ratio,  # Conservative for preservation
            
            # Data processing options
            'use_processed_dataset': use_processed_dataset,
            'unmask_messages': unmask_messages,
            
            # Optimization
            'use_liger': True,                     # Enable Liger kernels
            'seed': 0,
            
            # Learning rate scheduler
            'lr_scheduler': "cosine",              # Cosine scheduler works well with OSFT
        }
                
        # For single-GPU training, disable FSDP hybrid sharding
        # by using FULL_SHARD (2) instead of HYBRID_SHARD (default)
        if num_gpus == 1:
            self.training_kwargs['fsdp_sharding_strategy'] = 2  # FULL_SHARD


    def estimate(self, method="sft", num_epochs=1):
        start_time = time.time()
        if method == "osft": 
            # _ = osft(**self.training_kwargs)
            # warmup_time, final_time, time_per_iter, total_steps = self._osft_log_parse(os.path.join(self.training_kwargs['ckpt_output_dir'], 'training_log_node0.log'))
            warmup_time, final_time, time_per_iter, total_steps = self._osft_log_parse('training_log_node0.log')
        elif method == "lora":
            _ = lora_sft(**self.training_kwargs)
        else:
            _ = sft(**self.training_kwargs)
            warmup_time, final_time, time_per_iter, total_steps = self._sft_log_parse(os.path.join(self.training_kwargs['ckpt_output_dir'], 'full_logs_global0.log'))

        _, _, report = self.TimeReport(warmup_time, final_time, total_steps)     

        with open(os.path.join(self.training_kwargs['ckpt_output_dir'], 'time_estimate.log'), 'w') as f:
            f.write(report)

        end_time = time.time()
        duration = end_time - start_time


    def estimate_from_file(self, file_path: str, method):
        if method == "osft":
            warmup_time, final_time, time_per_iter, total_steps = self._osft_log_parse(file_path)
        else:
            warmup_time, final_time, time_per_iter, total_steps = self._sft_log_parse(file_path)

        _, _, report = self.TimeReport(warmup_time, final_time, total_steps)     

        with open(os.path.join(self.training_kwargs['ckpt_output_dir'], 'time_estimate.log'), 'w') as f:
            f.write(report)

    def _osft_log_parse(self, file_path: str):
        # TODO: Handle the date in the timestamp?
        line_list = []
        batch_time_list = []
        batch_count = 0
        diff_list = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.__contains__('timestamp'):
                    timestamp = line.split("\"")[-2].split('-')[-1].split('T')
                    day = int(timestamp[0])
                    clock_time = timestamp[1].split(':')
                    hour = int(clock_time[0])
                    minute = int(clock_time[1])
                    second = float(clock_time[2])
                    line_list.append(timedelta(days=day, hours=hour, minutes=minute, seconds=second))
                if line.__contains__("Epoch"):
                    target_str = line.split('/')[1].split('\x1b')[0]
                    batch_count = int(target_str)
                if line.__contains__("time_per_batch"):
                    batch_time_list.append(float(line.split('\x1b')[-2].split('m')[1]))
        for i in range(1, len(line_list)):
            time_diff = line_list[i] - line_list[i-1]
            raw_second_float = time_diff.seconds + time_diff.microseconds / 1000000
            diff = raw_second_float - batch_time_list[i]
            diff_list.append(diff)
        avg_batch_len = np.mean(batch_time_list[:-1])
        avg_extra_len = np.mean(diff_list[:-1])

        total_timedelta = line_list[-1] - line_list[0]
        total_time = (total_timedelta.seconds + total_timedelta.microseconds / 1000000) + batch_time_list[0] + avg_extra_len

        avg_total_len = avg_batch_len + avg_extra_len

        print(total_time)
        print(int(total_time / 60))
        print(total_time % 60)


        return 0, total_time, avg_total_len, batch_count 


    def TimeReport(self, warmup_time, final_time, total_steps):
        true_time_per_iter = (final_time - warmup_time) / total_steps
        remaining_steps = (total_steps / self.sample_frac) * self.remaining_frac

        additional_time = remaining_steps * true_time_per_iter
        subtotal_time = int(final_time + additional_time)
        total_time = int(subtotal_time * 1.1)

        sub_hrs = str(math.floor(subtotal_time / 3600))
        sub_min = str(math.floor(subtotal_time / 60) % 60) 
        if len(sub_min) == 1: sub_min = '0' + sub_min
        sub_sec = str(subtotal_time % 60)
        if len(sub_sec) == 1: sub_sec = '0' + sub_sec

        tot_hrs = str(math.floor(total_time / 3600))
        tot_min = str(math.floor(total_time / 60) % 60) 
        if len(tot_min) == 1: tot_min = '0' + tot_min
        tot_sec = str(total_time % 60)
        if len(tot_sec) == 1: tot_sec = '0' + tot_sec

        res_string = "Estimated Training Range In Hours:Minutes:Seconds: "  + \
            sub_hrs + ":" + sub_min + ":" + sub_sec + " — " + \
            tot_hrs + ":" + tot_min + ":" + tot_sec

        print(res_string)

        low_time = timedelta(hours=float(sub_hrs), minutes=float(sub_min), seconds=float(sub_sec))
        high_time = timedelta(hours=float(tot_hrs), minutes=float(tot_min), seconds=float(tot_sec))

        return low_time, high_time, res_string


    def _sft_log_parse(self, file_path: str):
        line_list = []
        with open(file_path, 'r') as f:
            for line in f:
                if line[:7] == 'Epoch 0':
                    line_list.append(line)
        warmup_step = line_list[1]
        final_step = line_list[-1]

        split_words_warmup = warmup_step.split('<')[0].split()
        split_words_final = final_step.split()
        time_per_iter = float(split_words_final[-1][:-5])
        warmup_time = split_words_warmup[-1][1:]
        final_time = split_words_final[-2].split('<')[0][1:]
        total_steps = int(warmup_step.split('/')[1].split()[0])

        def time_to_seconds(time_str: str):
            split_time = time_str.split(':')
            multiplier = 1
            total_time = 0
            for i in range(len(split_time) - 1, -1, -1):
                total_time += int(split_time[i]) * multiplier
                multiplier *= 60
            return total_time

        warmup_time_sec = time_to_seconds(warmup_time)
        final_time_sec = time_to_seconds(final_time)

        print(final_time_sec)
        print(int(final_time_sec / 60))
        print(final_time_sec % 60)

        return warmup_time_sec, final_time_sec, time_per_iter, total_steps

    def _LoRATimeParse(self):
        pass

    def _clean_artifacts(self):
        os.removedirs(self.training_kwargs['ckpt_output_dir'])
        os.remove('temp_data.jsonl')


class AdvancedFLOPTimingEstimator:
    """
    2203
    Embeddings: 2 × seq_len × vocab_size × d_model
    Attention (Single Layer):
    Key, query and value projections: 2 × 3 × seq_len × d_model × (key_size × num_heads)
    Key @ Query logits: 2 × seq_len × seq_len × (key_size × num_heads)
    Softmax: 3 × num_heads × seq_len × seq_len
    Softmax @ query reductions: 2 × seq_len × seq_len × (key_size × num_heads)
    Final Linear: 2 × seq_len × (key_size × num_heads) × d_model
    Dense Block (Single Layer): 2 × seq_len × (d_model × ffw_size + d_model × ffw_size)
    Final Logits: 2 × seq_len × d_model × vocab_size
    Total forward pass FLOPs: embeddings+num_layers× (total_attention+dense_block) + logits

    ffw_size: Feed Forward Weight Size = 'intermediate_size' in model.config
    key_size: Key Size: Based on the model architecture, check the first dim of layers.0.self_attn.k_proj.weight
    num_heads: Number of Heads: 'num_attention_heads' in model.config / 'num_key_value_heads' in model.config
    d_model: Model Dimension = 'hidden_size' in model.config
    vocab_size: Vocabulary Size: 'vocab_size' in model.config
    seq_len: Sequence Length
    """
    def __init__(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModel.from_pretrained(model_name)
        tensors = list(model.state_dict().values())
        tensor_sizes = [tensor.shape for tensor in tensors]
        model_dict = {key: value for key, value in zip(model.state_dict().keys(), tensor_sizes)}
        d_model = model.config.hidden_size # d_model
        num_heads = model.config.num_attention_heads
        # print(model.config["num_key_value_heads"])
        num_layers = model.config.num_hidden_layers
        vocab_size = model.config.vocab_size
        ffw_size = model.config.intermediate_size
        target_key = 'layers.0.' + MODEL_CONFIGS['qwen']['patterns'][1] + '.weight'
        key_size = model_dict[target_key][0]
        print(key_size)
        seq_len = 10000

        # TODO: Note that the sequence length is a separable variable from all of these...if we can determine
        # how many tokens are in the dataset, we should be able to determine the total FLOP count......
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset_path = "table_ds.jsonl"
        token_counts = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                tokens = tokenizer.encode(data['messages'][0]['content']) + tokenizer.encode(data['messages'][1]['content']) + tokenizer.encode(data['messages'][2]['content'])
                token_counts.append(len(tokens))

        final_tot_flops = 0
        for token_count in token_counts:
            seq_len = token_count

            embedding_flops = 2 * seq_len * vocab_size * d_model

            kqv_proj = 2 * 3 * seq_len * d_model * key_size * num_heads
            kq_logits = 2 * seq_len * seq_len * (key_size * num_heads)
            softmax = 3 * num_heads * seq_len * seq_len
            softmax_q = 2 * seq_len * seq_len * (key_size * num_heads)
            final_linear = 2 * seq_len * key_size * num_heads * d_model
            total_attention = kqv_proj + kq_logits + softmax + softmax_q + final_linear

            dense_block = 2 * seq_len * (d_model * ffw_size * 2) 
            logits = 2 * seq_len * d_model * vocab_size

            tot_flops = embedding_flops + (num_layers * (total_attention+dense_block)) + logits
            final_tot_flops += tot_flops

        print(final_tot_flops)

class BasicFLOPTimingEstimator:
    """
    2505/2401
    Training: 6PD_train (Parameters, Num Tokens)
    Inference/Pass Through: 2PD_inf (Parameters, Num Tokens)
    """
    def __init__(self):
        pass

    def estimate(self, model_params, tokens_per_gpu):
        pass
"""
if __name__ == "__main__":
    # llama_estimator = OnlineTimingEstimator("meta-llama/Llama-3.2-1B-Instruct")
    # llama_estimator.estimate('osft')
    test_types = ["osft"]
    datasets = ["table_ds" , "SDG", "long_ds"]
    models = [
        "google/gemma-3-1b-it",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "ibm-granite/granite-3.3-2b-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    for test_type in test_types:
        for dataset in datasets:
            for model in models:
                try:
                    ckpt_dir = "ckpt_" + test_type + "_8192_" + dataset + "_2_" + model.split('/')[1]
                    if test_type == "osft":
                        ckpt_dir += "_2"
                    print(ckpt_dir)
                    estimator = OnlineTimingEstimator(model, dataset_dir=dataset+".jsonl", ckpt_dir=ckpt_dir)
                    if test_type == "sft":
                            estimator._sft_log_parse(file_path=os.path.join('2-3-res', ckpt_dir, "full_logs_global0.log")) # , method="sft")
                    else:
                            estimator._osft_log_parse(file_path=os.path.join('2-3-res', ckpt_dir, "training_log_node0.log")) # , method="osft")
                except:
                    print("Skipping...")
"""
# AdvancedFLOPTimingEstimator()

dataset_path = pd.read_json(path_or_buf=os.path.join('2-4-timing-res', 'granite-3.3-2b-instruct_osft_table_ds', 'temp_data.jsonl'), lines=True)
tokenizer = AutoTokenizer.from_pretrained('granite-3.3-2b-instruct')
token_counts = []
with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                tokens = tokenizer.encode(data['messages'][0]['content']) + tokenizer.encode(data['messages'][1]['content']) + tokenizer.encode(data['messages'][2]['content'])
                token_counts.append(len(tokens))
print(token_counts)