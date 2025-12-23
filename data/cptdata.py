from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import torch


def _get_bin(split: str):
    bin_data_dir = '/new_data/wenlong/wz-entigraph-data/data/dataset/bins'
    bin_data_dir2 = 'data/dataset/bins'
    bin_data_dir3 = '/new_data/wenlong_rh/wz-entigraph-data/data/dataset/bins'
    splits = {
        # pretraining data replay
        'redpj-train':           f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
        'redpj-test':            f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_test.bin',
        'raw': f'{bin_data_dir}/quality_all-raw.bin',
        'human-pipeline-document_20%': f'{bin_data_dir}/quality_all-human-pipeline-document_20.bin',
        'human-pipeline-document_154': f'{bin_data_dir}/quality_all-human-pipeline-document_154.bin',
        'human-pipeline-document_154_dedup': f'{bin_data_dir}/quality_all-human-pipeline-document_154_dedup.bin',
        'human-pipeline-document_154_cqaqa': f'{bin_data_dir}/quality_all-human-pipeline-document_154_cqaqa.bin',
        'quality_adaptllm': f'{bin_data_dir}/quality_adaptllm.bin',
        'sdg_hub_knowledge_v0_6': f'{bin_data_dir}/sdg_hub_knowledge_v0_6.bin',
        'sdg_hub_knowledge_v0_6_1': f'{bin_data_dir}/sdg_hub_knowledge_v0_6_1.bin',
        'sdg_hub_knowledge_v0_6_4': f'{bin_data_dir}/sdg_hub_knowledge_v0_6_4.bin',
        'synth_knowledge2_0_xcombined_cut_100x_cpt_non_tokenized': f'{bin_data_dir}/synth_knowledge2_0_xcombined_cut_100x_cpt_non_tokenized.bin',
        'synth_knowledge2_0_xcombined_cut_50x_cpt_non_tokenized': f'{bin_data_dir}/synth_knowledge2_0_xcombined_cut_50x_cpt_non_tokenized.bin',
        'synth_knowledge2_0_xcombined_cut_30x_cpt_non_tokenized': f'{bin_data_dir}/synth_knowledge2_0_xcombined_cut_30x_cpt_non_tokenized.bin',
        # corpus task-agnostic augmentation
        # entigraph	                449,430,425
        # entigraph_26              39,953,072
        # entigraph_pruned_v1       18,459,386
        # entigraph_pruned_v1.1	    19,974,261
        # entigraph_pruned_v1.2	    36,137,842
        # official_qa_sft           8,627,371
        # flow_0.1                  7,613,627
        'entigraph':             f'{bin_data_dir}/quality_all-entigraphgpt-4-turbo.bin',
        'entigraph_26':          f'{bin_data_dir3}/quality_all-entigraph_26_gpt-4-turbo.bin',
        'entigraph_pruned_v1':   f'{bin_data_dir}/quality_all-entigraphgpt-4-turbo-pruned-v1.bin',
        'entigraph_pruned_v1.1': f'{bin_data_dir}/quality_all-entigraphgpt-4-turbo-pruned-v1.1.bin',
        'entigraph_pruned_v1.2': f'{bin_data_dir}/quality_all-entigraphgpt-4-turbo-pruned-v1.2.bin',

        'entigraph_pruned_subset_50': f'{bin_data_dir3}/quality_all-entigraphgpt-4-turbo-subset_50.0.bin',
        'entigraph_pruned_subset_75': f'{bin_data_dir3}/quality_all-entigraphgpt-4-turbo-subset_75.0.bin',
        'entigraph_pruned_subset_90': f'{bin_data_dir3}/quality_all-entigraphgpt-4-turbo-subset_90.0.bin',
        'entigraph_pruned_subset_95': f'{bin_data_dir3}/quality_all-entigraphgpt-4-turbo-subset_95.0.bin',
        'entigraph_pruned_subset_97': f'{bin_data_dir3}/quality_all-entigraphgpt-4-turbo-subset_97.0.bin',

        'official_qa_sft':       f'{bin_data_dir}/quality_qasftgpt-4-turbo.bin',  # in fact qa continued pretraining
        'official_qa_sft_mcq':       f'{bin_data_dir}/quality_qasft_mcqgpt-4-turbo.bin',
        'flow_0.1':              f'{bin_data_dir}/flow_0.1.bin',  # our generated qa continued pretraining data

        # corpus instruction tuning data
        # TODO

        # IL_qasft_v1
        'IL_qasft_v1_train':       f'IL_qasft_v1_train.bin',

        # generic instruction tuning data
        'ultrachat-train':       f'{bin_data_dir3}/ultrachat_train.bin',
        'ultrachat-train-mcq':       f'{bin_data_dir3}/ultrachat_train_mcq.bin',
        'ultrachat-test':        f'{bin_data_dir2}/ultrachat_test.bin'
    }
    print(splits.keys())
    assert split in splits, f"{split} is not implemented."
    return splits[split]


class _MemmapDataset(Dataset):
    def __init__(self, split: str, bin_file: str, subsample_ratio: float, block_size: int):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode='r')
        input_ids_size = len(self.ids)
        print(f"{split} has {input_ids_size} tokens.")
        self.ids = self.ids[:int(len(self.ids) * subsample_ratio)]
        if subsample_ratio < 1.0:
            print(f"Subsampled {subsample_ratio} to get {len(self.ids)} tokens.")

    def __len__(self):
        return int(len(self.ids) / self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i * self.block_size
        end_ind = (i + 1) * self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        return dict(
            input_ids=torch.from_numpy(x_id).long(),
            labels=torch.from_numpy(x_id).long()
        )


class CPTDataset(_MemmapDataset):
    def __init__(
        self, split: str, subsample_ratio: float, block_size: int, rehersal_rate: float
    ):  # TODO: multiple splits and subsample ratios
        assert rehersal_rate <= 1.0
        self.rehersal_rate = rehersal_rate
        self.rehersal_data = _MemmapDataset('redpj-train', _get_bin('redpj-train'), 1.0, block_size)
        super().__init__(split, _get_bin(split), subsample_ratio, block_size)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if np.random.rand() < self.rehersal_rate:
            idx = np.random.randint(len(self.rehersal_data))
            return self.rehersal_data[idx]
        else:
            return super().__getitem__(i)


def get_task_data_module(
    task_name: str,
    split: str,
    subsample_ratio: float,
    block_size: int,
    rehersal_rate: float,  # TODO: allow an array if there are multiple training datasets
    **kwargs
):
    if task_name == 'quality':
        train = CPTDataset(split, subsample_ratio, block_size, rehersal_rate)
        val = _MemmapDataset('redpj-test', _get_bin('redpj-test'), 1.0, block_size)  # this is grabbing the 'red pajamas'
        return dict(train_dataset=train, eval_dataset=val)
    if task_name == 'instruct':
        train = _MemmapDataset(split, _get_bin(split), subsample_ratio, block_size)
        val = _MemmapDataset('ultrachat-test', _get_bin('ultrachat-test'), subsample_ratio, block_size)
        return dict(train_dataset=train, eval_dataset=val)
    raise NotImplementedError(f"Task {task_name} is not implemented")


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    tokenizer.model_max_length = 2 ** 20  # this is to hide the token_len>128K wraning

    block_size = 2048
    rehersal_rate = 0.1
    subsample_ratio = 1.0
    task_name = 'quality'
    split = 'entigraph'
    data_module = get_task_data_module(task_name, split, subsample_ratio, block_size, rehersal_rate)
    for example in data_module['train_dataset']:
        print(tokenizer.decode(example['input_ids']))
        import pdb; pdb.set_trace()
