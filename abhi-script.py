"""
Each input document is a .json file that saves:
[[entity1, entity2, ...], discussion1, discussion2, ...]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
from tqdm import tqdm
from datasets import load_dataset

DATA_DIR = "/new_data/wenlong/wz-entigraph-data/data/dataset/raw"
BIN_DIR = "/new_data/wenlong/wz-entigraph-data/data/dataset/bins"

# /new_data/knowledge_rh/quality/training_mix/exp-2-x/level1_level2_level3_v2_test_set_qa3_d_synth_percent_154%.jsonl
def _get_quality_graph(model_name: str) -> List[str]:
    ds = load_dataset('json', data_files="/new_data/knowledge_rh/quality/synth_knowledge2_0/training_mix/combined_cut_100x_cpt_non_tokenized.jsonl", split='train')
    ds = ds.map(lambda x: {'document': f"{x['document_outline']}\n\n{x['document']}"}, num_proc=10)
    return list(set(ds['document']))


def get_tokenizer(tokenizer_model_name: str) -> AutoTokenizer:
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length = 2 ** 20  # this is to hide the token_len>128K warning
    return tokenizer


def tokenize_list(text_list: List[str]) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct") #"meta-llama/Meta-Llama-3-8B"
    all_ids = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text)  # by default, add_special_tokens=True adds the BOS token
            ids.append(tokenizer.eos_token_id)  # add the end of text token
            all_ids.extend(ids)
    print(len(all_ids))
    return all_ids


def write_to_memmap_single(ids: List[int], filename: str):
    filename = f'{BIN_DIR}/{filename}'
    print(f'Writing to {filename} with length {len(ids)}')
    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()


def tokenize_quality_graph(model_name: str):
    quality = _get_quality_graph(model_name)
    write_to_memmap_single(tokenize_list(quality), f'synth_knowledge2_0_xcombined_cut_100x_cpt_non_tokenized.bin')


if __name__ == '__main__':
    tokenize_quality_graph('gpt-4-turbo')
