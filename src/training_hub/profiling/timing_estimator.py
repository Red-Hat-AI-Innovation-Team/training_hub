import warnings
import os
import shutil
import pandas as pd
import math
import numpy as np
from datetime import timedelta

from training_hub import sft, osft, lora_sft

"""
Code assisted by Cursor/Claude4
This code is based on the scripts in the examples/scripts folder. 
"""

# TODO: Make sure to adjust the scale correction by basing it on the remaining TOKENS in the dataset
# (this calculation should only take a few seconds...maybe this too can be cached?)

class TimingEstimatorExperimental:
    """
    An experimental class for estimating the runtime for a fine-tuning process.

    Args (in addition to the standard training kwargs):
        model_path (str): HuggingFace model path to the model to fine-tune
        num_gpus (int): Number of GPUs to use for training (default: 2)
        dataset_dir (str): The path to the dataset to use for the estimation
        sample_frac (float): The fraction of the dataset that will be used for the estimation
        ckpt_dir (str): The name of thedirectory to save the data to. It will be created if it doesn't exist 
        clean_up_artifacts (bool): If set to True, delete all files generated during training.
                                    **This will delete the contents of ckpt_dir!**
        verbose (int): The level of verbosity to print out. Set to 0 for no printing,
                        set to 1 to print out results to console.
    """
    def __init__(self,
                model_path: str,
                num_gpus: int=2,
                dataset_dir: str="table_ds.jsonl",
                sample_frac: float=0.05,
                ckpt_dir: str = 'timing_temp_ckpts',
                clean_up_artifacts=False,
                verbose=1,
                batch_size: int=128,
                max_seq_len: int=8192,
                max_tokens_per_gpu: int=10000,
                lr: float=1e-5,
                unfreeze_rank_ratio: float=0.2,
                use_processed_dataset=False,
                unmask_messages=False,
                ):
    
        warnings.warn("This is an experimental class for timing estimation. The proposed estimates may vary greatly from your actual time.")

        # Store the relevant parameters
        self.sample_frac = sample_frac
        self.remaining_frac = 1.0 - self.sample_frac
        self.clean_up_artifacts = clean_up_artifacts
        self.verbose = verbose

        # Create a temp jsonl that only has a certain portion of the data
        self.form_data_subset(dataset_dir)
        
        # Create a temporary directory to save checkpoints too 
        try: os.mkdir(ckpt_dir)
        except FileExistsError: pass

        # **Based on the example code from /examples/scripts**
        # Set the fine-tuning arguments: 
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


    def form_data_subset(self, dataset_dir: str):
        """
        Helper function to form the data subset that the estimation will be based on
        It's a separate function in case you want to form new data later.
        """
        data = pd.read_json(path_or_buf=dataset_dir, lines=True)
        sampled_data = data.sample(frac=self.sample_frac, axis=0)
        self.num_samples = len(sampled_data)
        sampled_data.to_json(path_or_buf='temp_data.jsonl', lines=True, orient='records')


    def estimate(self, method: str="sft", num_epochs: int=1):
        """
        Perform the timing estimation.

        Args:
            method (str): The method to use for timing estimation. (Note, Only "sft" and "osft" are supported). Defaults to sft.
            num_epochs (int): The number of epochs to use for full timing estimation (the subset training will only run for 1 epoch)

        Returns:
            low_time (timedelta): The lower bound of the estimated time.
            high_time (timedelta): The upper bound of the estimated time.        
        """
        # With the given training method, fine-tune the pretrained model on the data subset,
        # then use the logs to extrapolate a time estimate.
        if method == "osft": 
            _ = osft(**self.training_kwargs)
            low_time, high_time = self.estimate_from_file(os.path.join(self.training_kwargs['ckpt_output_dir'], 'training_log_node0.log'), method, num_epochs)
        elif method == "lora":
            raise NotImplementedError("LORA timing estimation is not implemented yet.")
        elif method == "qlora":
            raise NotImplementedError("QLORA timing estimation is not implemented yet.")
        else:
            print("Defaulting to SFT")
            _ = sft(**self.training_kwargs)
            low_time, high_time = self.estimate_from_file(os.path.join(self.training_kwargs['ckpt_output_dir'], 'training_log_node0.log'), method, num_epochs)

        # Delete all files in this training process once the estimation is complete. 
        if self.clean_up_artifacts: self._clean_artifacts()

        return low_time, high_time


    def estimate_from_file(self, file_path: str, method, num_epochs=1):
        """
        Use a fine-tuning log to estimate the time needed for a full training run. 
        Note that LoRA and QLoRA logs currently aren't supported. 

        Args:
            file_path (str): The path to the log file.
        """
        # Parse the log files
        if method == "osft": warmup_time, final_time, time_per_iter, total_steps = self._osft_log_parse(file_path)
        elif method == "lora": raise NotImplementedError("LORA timing estimation is not implemented yet.")
        elif method == "qlora": raise NotImplementedError("QLORA timing estimation is not implemented yet.")
        else: warmup_time, final_time, time_per_iter, total_steps = self._sft_log_parse(file_path)
        
        # Find the lower and upper bound times for a single epoch
        if self.verbose > 0:
            print("Time estimation for 1 epoch:")
        low_time, high_time = self.time_report(warmup_time, final_time, total_steps)

        # If the input calls for multiple epochs, find the lower and upper bounds
        # for multipl epochs too
        if num_epochs > 1:
            if self.verbose > 0:
                print("Time estimation for " + str(num_epochs) + " epochs:")
            low_time, high_time = self.time_report(warmup_time, final_time * num_epochs, total_steps * num_epochs)  

        return low_time, high_time


    def _osft_log_parse(self, file_path: str):
        # TODO: Handle the date in the timestamp?
        """
        Parse the SFT log file

        Returns:
            warmup_time (float): The time needed before training begins.
            total_time (float): The time needed for the full raining process
            avg_total_len (float): The average time needed per step.
            batch_count (int): The number of training batches that were used. 
        """
        # Iterate through the entire file
        line_list = []
        batch_time_list = []
        batch_count = 0
        diff_list = []

        with open(file_path, 'r') as f:
            for line in f:
                # Store the current step and timestamp
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

        # Find the average time that passed during each training step.
        # Specifically, take the difference between the timestamps and measure how 
        # they differ between the actual reporting training time to find the
        # additional overhead time.
        for i in range(1, len(line_list)):
            time_diff = line_list[i] - line_list[i-1]
            raw_second_float = time_diff.seconds + time_diff.microseconds / 1000000
            diff = raw_second_float - batch_time_list[i]
            diff_list.append(diff)
        avg_batch_len = np.mean(batch_time_list[:-1])
        avg_extra_len = np.mean(diff_list[:-1])
        avg_total_len = avg_batch_len + avg_extra_len

        # Get the total length of time that was needed for this fine-tuning subset
        total_timedelta = line_list[-1] - line_list[0]
        total_time = (total_timedelta.seconds + total_timedelta.microseconds / 1000000) + batch_time_list[0] + avg_extra_len

        return 0, total_time, avg_total_len, batch_count 


    def _sft_log_parse(self, file_path: str):
        """
        Parse the OSFT log file

        Returns:
            warmup_time (float): The time needed before training begins.
            total_time (float): The time needed for the full raining process
            avg_total_len (float): The average time needed per step.
            batch_count (int): The number of training batches that were used. 
        """
        # Collect all logged times needed for each batch
        line_list = []
        with open(file_path, 'r') as f:
            for line in f:
                if line[:7] == 'Epoch 0':
                    line_list.append(line)

        # The warmup step is measured as the second line, 
        # because the first line declares the start of training.
        warmup_step = line_list[1]
        final_step = line_list[-1]

        # Parse the warmup line to find the amount of time that was needed
        # for the warmup step, and the total steps for training
        split_words_warmup = warmup_step.split('<')[0].split()
        warmup_time = split_words_warmup[-1][1:]
        total_steps = int(warmup_step.split('/')[1].split()[0])

        # Do the same for the final time.
        split_words_final = final_step.split()
        time_per_iter = float(split_words_final[-1][:-5])
        final_time = split_words_final[-2].split('<')[0][1:]

        # Convert each of these times into total seconds
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

        return warmup_time_sec, final_time_sec, time_per_iter, total_steps


    def _LoRATimeParse(self):
        pass


    def _clean_artifacts(self):
        """
        Clean out any files generated during training. 
        """
        shutil.rmtree(self.training_kwargs['ckpt_output_dir'])
        os.remove('temp_data.jsonl')


    def _time_strings_from_sec(self,subtotal_time: int):
        """
        Helper function to convert the time in seconds to hrs/min/sec strings
        """
        sub_hrs = str(math.floor(subtotal_time / 3600))
        sub_min = str(math.floor(subtotal_time / 60) % 60) 
        if len(sub_min) == 1: sub_min = '0' + sub_min
        sub_sec = str(subtotal_time % 60)
        if len(sub_sec) == 1: sub_sec = '0' + sub_sec
        return sub_hrs, sub_min, sub_sec

    def time_report(self, warmup_time, final_time, total_steps):
        """
        Report the estimated time needed for the full fine-tuning process.

        Args:
            warmup_time (float): The time needed before training begins.
            avg_total_len (float): The average time needed per step.
            total_steps (int): The number of steps in the training process.

        Returns:
            low_time (timedelta): The lower bound of the estimated time.
            high_time (timedelta): The upper bound of the estimated time.
        """
        # Find the time needed for the sampled training process, besides the warmup
        true_time_per_iter = (final_time - warmup_time) / total_steps
        remaining_steps = (total_steps / self.sample_frac) * self.remaining_frac

        # Calculate the expected time needed to do the rest of the training
        additional_time = remaining_steps * true_time_per_iter

        # Use this value to create low and high bounds for the time needed
        subtotal_time = int(final_time + additional_time)
        total_time = int(subtotal_time * 1.1)

        # Convert each of these times into hrs/min/sec
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

        res_string = "Estimated Training Range In Hours:Minutes:Seconds for 1 epoch: "  + \
            sub_hrs + ":" + sub_min + ":" + sub_sec + " — " + \
            tot_hrs + ":" + tot_min + ":" + tot_sec

        low_time = timedelta(hours=float(sub_hrs), minutes=float(sub_min), seconds=float(sub_sec))
        high_time = timedelta(hours=float(tot_hrs), minutes=float(tot_min), seconds=float(tot_sec))

        if self.verbose > 0: print(res_string)

        return low_time, high_time


def time_estimate(method: str,
                model_path: str,
                num_gpus: int=2,
                dataset_dir: str="table_ds.jsonl",
                batch_size: int=128,
                max_seq_len: int=8192,
                max_tokens_per_gpu: int=10000,
                lr: float=1e-5,
                unfreeze_rank_ratio: float=0.2,
                sample_frac: float=0.05,
                ckpt_dir: str = 'timing_temp_ckpts',
                use_processed_dataset=False,
                unmask_messages=False,
                num_epochs: int=1,
                clean_up_artifacts: bool=True,
                verbose: int=1
                ):
    """
    Convenience function for estimating the time needed for a full fine-tuning process
    """

    estimator = TimingEstimatorExperimental(model_path=model_path,
                num_gpus=num_gpus,
                dataset_dir=dataset_dir,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                max_tokens_per_gpu=max_tokens_per_gpu,
                lr=lr,
                unfreeze_rank_ratio=unfreeze_rank_ratio,
                sample_frac=sample_frac,
                ckpt_dir=ckpt_dir,
                use_processed_dataset=use_processed_dataset,
                unmask_messages=unmask_messages,
                clean_up_artifacts=clean_up_artifacts,
                verbose=verbose)

    low_time, high_time = estimator.estimate(method=method, num_epochs=num_epochs)

    return low_time, high_time


def time_estimate_from_file(file_path: str, method: str, sample_frac: float=0.05, num_epochs: int=3, verbose: int=1):
    """
    Convenience function for estimating the time needed for a full fine-tuning process, based on a given log file
    """

    estimator = TimingEstimatorExperimental(model_path="",
                sample_frac=sample_frac,
                verbose=verbose)

    low_time, high_time = estimator.estimate_from_file(file_path=file_path, method=method, num_epochs=num_epochs)

    return low_time, high_time
