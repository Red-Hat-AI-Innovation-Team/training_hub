# Runtime Example Measurements

Below are example wall clock measurements for how long you can expect a model to take to be fine-tuned. 

AI Use Disclosure: Please note that all bar plots shown in this document are based on code primarily generated via Cursor/Claude 4 (with human oversight and manual adjustments) and the script for finding the number of samples and tokens in the datasets was written with Cursor/Claude 4 assistance. 

## General Experiment Notes
- The experiments were conducted by using the default settings provided by `sft_granite_example.py` and `osft_granite_example.py`
    - For SFT, `max_tokens_per_gpu=25000` and `max_seq_len=20000`
    - For OSFT, `max_tokens_per_gpu=10000`, `max_seq_len=4096`, and `unfreeze_rank_ratio=0.3`
- **Models**: Two models were tested, **Granite 3.3 8B**, and **Granite 4 Tiny Preview** (a Mixture-of-Experts model that also has 8B Parameters)
- **Hardware**: Two different hardware configurations were tested, a server with **8x A100s**, and an Openshift cluster with **8x H100s**. 
- **Datasets**: Two datasets were tested, a simple dataset in Table-GPT and a much larger and longer dataset in Bespoke-Stratos-17k.
    - Please note that both datasets were obtained by downloading the dataset from HuggingFace and then extracting the .jsonl file. 
- All experiments were run for the first full epoch two times, with the displayed time being the average of the two times. 
    - **Please be aware that time for later epochs may vary**
    - On the A100 machine, the variation between the two runs was negligible, never more than 6 seconds. 
    - The variation is a bit larger on the H100 machine, especially during the first run of a pod (the first result was discarded and reran if it varied significantly)
    - **Granite 4 tends to require some amount of warm-up before its first usage.** The shown times for Granite 4 are for runs where the warm-up does not occur. Typically, the warm-up adds about an additional 1 minute to the runtime.
        - The reasons for and details about this warm-up aren't known at the moment, but will be added when more information is gathered. Please keep in mind Granite 4 is still in preview. 
- The time measurement is calculated by using the timestamps logged during the training process in the above scripts
- By default, OSFT makes use of Liger Kernels to improve memory usage and runtime. However, as of Nov 7th 2025, Liger Kernels currently don't have built-in support for Granite 4
    - As a result, the script was modified for allow Liger Kernels to be disabled for certain experiments
    - The tables will be updated once support for Liger Kernels is added. 
- Many of these tests had the checkpointing hardcoded to be disabled in the script (set `checkpoint_at_epoch=False` and `accelerate_full_state_at_epoch=False`)
    - This does not appear to impact runtime of the actual training loop
    - This was mostly done to conserve disk space due to checkpoints being very large (tens of GB per epoch), which can cause DiskPressure on OpenShift

## Per Epoch Results

The following table shows the amount of time needed to run the first epoch of training

| Hardware | Training Type    | Model       | Dataset   | Train Time per Epoch (HH:MM:SS) |
| ---------| ---------------- | ----------- | --------- | ------------------------------- |
| 8x A100s | SFT              | Granite 3.3 | Table-GPT | 00:10:01                        | 
| 8x A100s | SFT              | Granite 3.3 | Bespoke   | 01:17:02                        | 
| 8x A100s | SFT              | Granite 4   | Table-GPT | 00:07:35                        | 
| 8x A100s | SFT              | Granite 4   | Bespoke   | 00:42:48                        | 
| 8x A100s | OSFT             | Granite 3.3 | Table-GPT | 00:36:09                        | 
| 8x A100s | OSFT             | Granite 3.3 | Bespoke   | 00:58:39                        | 
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | Table-GPT | 00:37:32                        | 
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | Bespoke   | 01:00:48                        | 
| 8x A100s | OSFT (No Liger)  | Granite 4   | Table-GPT | 00:14:11                        | 
| 8x A100s | OSFT (No Liger)  | Granite 4   | Bespoke   | 00:23:01                        | 
| 8x H100s | SFT              | Granite 3.3 | Table-GPT | 00:04:35                        | 
| 8x H100s | SFT              | Granite 3.3 | Bespoke   | 00:35:47                        | 
| 8x H100s | SFT              | Granite 4   | Table-GPT | 00:05:40                        | 
| 8x H100s | SFT              | Granite 4   | Bespoke   | 00:26:19                        | 
| 8x H100s | OSFT             | Granite 3.3 | Table-GPT | 00:46:04                        | 
| 8x H100s | OSFT             | Granite 3.3 | Bespoke   | 01:15:08                        | 
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Table-GPT | 00:46:51                        | 
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Bespoke   | 01:16:21                        | 
| 8x H100s | OSFT (No Liger)  | Granite 4   | Table-GPT | 00:10:39                        | 
| 8x H100s | OSFT (No Liger)  | Granite 4   | Bespoke   | 00:16:46                        | 


## Granite 3.3 vs Granite 4

Granite 3.3 is an open source **8B Parameter Large Language** Instruct model https://huggingface.co/ibm-granite/granite-3.3-8b-instruct
Granite 4 is still in preview stages, for these runs we use Tiny Preview, which is an open source **7B Parameter Hybrid Mixture-of-Experts** Instruct Model https://huggingface.co/ibm-granite/granite-4.0-tiny-preview 

### Graphs

![A graph comparing Granite 3.3 with Granite 4 for SFT][sftm]

[sftm]: https://github.com/mazam-lab/training_hub/blob/main/examples/docs/sft_models.png?raw=true ""

![A graph comparing Granite 3.3 with Granite 4 OSFT][osftm]

[osftm]: https://github.com/mazam-lab/training_hub/blob/main/examples/docs/osft_models.png?raw=true ""

### Tabled differences

| Hardware | Training Type    | Dataset   | Granite 3.3 Time | Granite 4 Time | Difference in Time | Multiplier Gain |
| -------- | ---------------- | --------- | ---------------- | -------------- | ------------------ | --------------- |
| 8x A100s | SFT              | Table-GPT | 00:10:01         | 00:07:35       | 00:02:26           | 0.76x           |                 
| 8x A100s | SFT              | Bespoke   | 01:17:02         | 00:42:48       | 00:34:14           | 0.55x           |
| 8x A100s | OSFT (No Liger)  | Table-GPT | 00:37:32         | 00:14:11       | 00:23:21           | 0.38x           |
| 8x A100s | OSFT (No Liger)  | Bespoke   | 01:00:48         | 00:23:01       | 00:37:47           | 0.38x           |
| 8x H100s | SFT              | Table-GPT | 00:04:35         | 00:05:40       | 00:01:05           | 1.24x           |
| 8x H100s | SFT              | Bespoke   | 00:35:47         | 00:26:19       | 00:09:28           | 0.74x           |
| 8x H100s | OSFT (No Liger)  | Table-GPT | 00:46:51         | 00:10:39       | 00:36:12           | 0.23x           |
| 8x H100s | OSFT (No Liger)  | Bespoke   | 01:16:21         | 00:16:46       | 00:59:35           | 0.22x           |


## A100 vs H100

The A100 and H100 both contain 80 GB of VRAM per GPUs, and both setups contain 8 GPUs. However, the H100 higher FLOPs and more cores than the A100. See Wikipedia: https://en.wikipedia.org/wiki/Hopper_(microarchitecture)#H100_accelerator_and_DGX_H100 

As mentioned, please note that the A100 setup is on an SSHed machine running containers, while the H100 setup is an OpenShift cluster. 


### Tabled differences

| Training Type   | Model       | Dataset   | A100 Time | H100 Time | Difference in Time | Multiplier Gain |
| --------------- | ----------- | --------- | --------- | --------- | ------------------ | --------------- |
| SFT             | Granite 3.3 | Table-GPT | 00:10:01  | 00:04:35  | 00:05:26           | 0.46x           |
| SFT             | Granite 3.3 | Bespoke   | 01:17:02  | 00:35:47  | 00:41:15           | 0.46x           |
| SFT             | Granite 4   | Table-GPT | 00:07:35  | 00:05:40  | 00:01:55           | 0.75x           |
| SFT             | Granite 4   | Bespoke   | 00:42:48  | 00:26:19  | 00:16:29           | 0.61x           |
| OSFT            | Granite 3.3 | Table-GPT | 00:36:09  | 00:46:04  | 00:09:55           | 1.27x           |
| OSFT            | Granite 3.3 | Bespoke   | 00:58:39  | 01:15:08  | 00:16:29           | 1.28x           |
| OSFT (No Liger) | Granite 3.3 | Table-GPT | 00:37:32  | 00:46:51  | 00:09:19           | 1.25x           |
| OSFT (No Liger) | Granite 3.3 | Bespoke   | 01:00:48  | 01:16:21  | 00:15:33           | 1.26x           |
| OSFT (No Liger) | Granite 4   | Table-GPT | 00:14:11  | 00:10:39  | 00:03:32           | 0.75x           |
| OSFT (No Liger) | Granite 4   | Bespoke   | 00:23:01  | 00:16:46  | 00:06:15           | 0.73x           |


## Table-GPT vs Bespoke

**Please keep in mind that for the SFT example `max_seq_len=20000` and for the OSFT example `max_seq_len=4096`.**

The two datasets used were Table-GPT (https://huggingface.co/datasets/LipengCS/Table-GPT) and Bespoke-Stratos-17k (https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k). Both datasets are using the training split. Each dataset was downloaded via Huggingface's datasets package, with the .jsonl file extracted for use in Training-Hub.

Please note that additional preprocessing was applied to the Bespoke dataset in order for Training-Hub to read it properly:
- The `conversations` column is renamed to `messages`
- The `system` column is renamed to `prompt`
- The `from` columns are renamed to `role`
- The `value` columns are renamed to `content`
- When `to_json` is applied, `compression=None`; there are some issues with reading the .jsonl otherwise.

### Statistics of Table-GPT and Bespoke's sample and token counts

The tokens of a given string are obtained using the tokenizers obtained from Granite 3.3 and Granite 4 using `AutoTokenizer`.

The number of tokens in each sample is the sum of the tokens contained within:
- The system prompt string
- The user prompt string
- The response string 

Due to differences in the tokenizer each model uses, Bespoke has slightly different token distributions when loaded with Granite 3.3 vs Granite 4. The differences between Granite 3.3 and Granite 4's statistics for Table-GPT were negligible. 

| Stat                 | Table-GPT   | Granite 3.3 Bespoke | Granite 4 Bespoke | 
| -------------------- | ----------- | ------------------- | ----------------- |
| # of Samples         | 13,222      | 16,710              | 16,710            |
| Lowest Token Length  | 135         | 640                 | 639               |
| Mean Token Length    | 931.25      | 6,023.17            | 6,011.41          |
| Token Length std     | 802.26      | 5,354.54            | 5,344.64          |
| Highest Token Length | 4,595       | 45,563              | 45,562            |


### Tabled differences

| Hardware | Training Type    | Model       | Table-GPT Time | Bespoke Time | Difference in Time | Multiplier |
| -------- | ---------------- | ----------- | -------------- | ------------ | ------------------ | ---------- |
| 8x A100s | SFT              | Granite 3.3 | 00:10:01       | 01:17:02     | 01:07:01           | 7.69x      |
| 8x A100s | SFT              | Granite 4   | 00:07:35       | 00:42:48     | 00:35:13           | 5.64x      |
| 8x A100s | OSFT             | Granite 3.3 | 00:36:09       | 00:58:39     | 00:22:30           | 1.62x      |
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | 00:37:32       | 01:00:48     | 00:23:16           | 1.62x      |
| 8x A100s | OSFT (No Liger)  | Granite 4   | 00:14:11       | 00:23:01     | 00:08:50           | 1.62x      |
| 8x H100s | SFT              | Granite 3.3 | 00:04:35       | 00:35:47     | 00:31:12           | 7.81x      |
| 8x H100s | SFT              | Granite 4   | 00:05:40       | 00:26:19     | 00:20:39           | 4.64x      |
| 8x H100s | OSFT             | Granite 3.3 | 00:46:04       | 01:15:08     | 00:29:04           | 1.63x      |
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | 00:46:51       | 01:16:21     | 00:29:30           | 1.63x      |
| 8x H100s | OSFT (No Liger)  | Granite 4   | 00:10:39       | 00:16:46     | 00:06:07           | 1.57x      |


## SFT vs OSFT 

As mentioned above, the experiments were conducted by using the default settings provided by `sft_granite_example.py` and `osft_granite_example.py`. For SFT, `max_tokens_per_gpu=25000` and `max_seq_len=20000`, while for OSFT, `max_tokens_per_gpu=10000`, `max_seq_len=4096`, and `unfreeze_rank_ratio=0.3`. Please note that not only does `max_tokens_per_gpu` have an impact on the time due to it effectively serving as a batch size, but `max_seq_len` prevents sequences exceeding that length from being processed. For a study on how SFT and OSFT compare against each other when `max_tokens_per_gpu` and `max_seq_len` are the same, see the **SFT Vs OSFT with the Same max_seq_len and max_tokens_per_gpu Values** subsection. 

For further details on OSFT and how it differs from SFT, please consult [Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning (Nayak et. al)](https://arxiv.org/abs/2504.07097)

### Liger Kernel Comparisons

| Hardware | Model       | Dataset   | SFT Time per Epoch | OSFT with Liger Time per Epoch | Difference in Time | Multiplier |
| ---------| ----------- | --------- | ------------------ | ------------------------------ | ------------------ | ---------- |
| 8x A100s | Granite 3.3 | Table-GPT | 00:10:01           | 00:36:09                       | 00:26:08           | 3.61x      |      
| 8x A100s | Granite 3.3 | Bespoke   | 01:17:02           | 00:58:39                       | 00:18:23           | 0.76x      |
| 8x H100s | Granite 3.3 | Table-GPT | 00:04:35           | 00:46:04                       | 00:41:29           | 10.05x     |
| 8x H100s | Granite 3.3 | Bespoke   | 00:35:47           | 01:15:08                       | 00:39:21           | 2.10x      | 

### No Liger Comparisons

| Hardware | Model       | Dataset   | SFT Time per Epoch | Adjusted OSFT (No Liger) Time per Epoch | Difference in Time | Multiplier |
| ---------| ----------- | --------- | ------------------ | --------------------------------------- | ------------------ | ---------- |
| 8x A100s | Granite 3.3 | Table-GPT | 00:10:01           | 00:37:32                                | 00:27:31           | 3.75x      |      
| 8x A100s | Granite 3.3 | Bespoke   | 01:17:02           | 01:00:48                                | 00:16:14           | 0.79x      |
| 8x A100s | Granite 4   | Table-GPT | 00:07:35           | 00:14:11                                | 00:07:36           | 1.87x      |
| 8x A100s | Granite 4   | Bespoke   | 00:42:48           | 00:23:01                                | 00:19:47           | 0.54x      |
| 8x H100s | Granite 3.3 | Table-GPT | 00:04:35           | 00:46:51                                | 00:42:16           | 10.22x     |
| 8x H100s | Granite 3.3 | Bespoke   | 00:35:47           | 01:16:21                                | 00:40:34           | 2.13x      | 
| 8x H100s | Granite 4   | Table-GPT | 00:05:40           | 00:10:39                                | 00:04:59           | 1.88x      |
| 8x H100s | Granite 4   | Bespoke   | 00:26:19           | 00:16:46                                | 00:09:33           | 0.64x      |


### SFT Vs OSFT with the Same max_seq_len and max_tokens_per_gpu Values

As noted above, many of Bespoke's samples exceed `max_seq_len=4096` used in `osft_granite_example.py`, but do not exceed `max_seq_len=20000` used in `sft_granite_example.py`. While we expect users to use the default provided values for each example script, for the sake of a fairer comparison between SFT and OSFT, some additional tests showing how OSFT fares with the higher `max_seq_len=20000` and `max_tokens_per_gpu=25000` values used in `sft_granite_example.py`. The tests with these adjusted values will be referred to as Adjusted OSFT. 

#### Full Results with adjusted OSFT

| Hardware | Training Type             | Model       | Dataset   | Train Time per Epoch (HH:MM:SS) |
| ---------| ------------------------- | ----------- | --------- | ------------------------------- |
| 8x A100s | SFT                       | Granite 3.3 | Table-GPT | 00:10:01                        | 
| 8x A100s | SFT                       | Granite 3.3 | Bespoke   | 01:17:02                        | 
| 8x A100s | SFT                       | Granite 4   | Table-GPT | 00:07:35                        | 
| 8x A100s | SFT                       | Granite 4   | Bespoke   | 00:42:48                        | 
| 8x A100s | OSFT                      | Granite 3.3 | Table-GPT | 00:36:09                        | 
| 8x A100s | OSFT                      | Granite 3.3 | Bespoke   | 00:58:39                        | 
| 8x A100s | OSFT (No Liger)           | Granite 3.3 | Table-GPT | 00:37:32                        | 
| 8x A100s | OSFT (No Liger)           | Granite 3.3 | Bespoke   | 01:00:48                        | 
| 8x A100s | OSFT (No Liger)           | Granite 4   | Table-GPT | 00:14:11                        | 
| 8x A100s | OSFT (No Liger)           | Granite 4   | Bespoke   | 00:23:01                        | 
| 8x A100s | Adjusted OSFT             | Granite 3.3 | Table-GPT | **00:34:23**                    | 
| 8x A100s | Adjusted OSFT             | Granite 3.3 | Bespoke   | **04:01:59**                    | 
| 8x A100s | Adjusted OSFT (No Liger)  | Granite 3.3 | Table-GPT | **00:35:42**                    | 
| 8x A100s | Adjusted OSFT (No Liger)  | Granite 3.3 | Bespoke   | **04:11:20**                    | 
| 8x A100s | Adjusted OSFT (No Liger)  | Granite 4   | Table-GPT | **00:09:06**                    | 
| 8x A100s | Adjusted OSFT (No Liger)  | Granite 4   | Bespoke   | **00:51:48**                    | 
| 8x H100s | SFT                       | Granite 3.3 | Table-GPT | 00:04:35                        | 
| 8x H100s | SFT                       | Granite 3.3 | Bespoke   | 00:35:47                        | 
| 8x H100s | SFT                       | Granite 4   | Table-GPT | 00:05:40                        | 
| 8x H100s | SFT                       | Granite 4   | Bespoke   | 00:26:19                        | 
| 8x H100s | OSFT                      | Granite 3.3 | Table-GPT | 00:46:04                        | 
| 8x H100s | OSFT                      | Granite 3.3 | Bespoke   | 01:15:08                        | 
| 8x H100s | OSFT (No Liger)           | Granite 3.3 | Table-GPT | 00:46:51                        | 
| 8x H100s | OSFT (No Liger)           | Granite 3.3 | Bespoke   | 01:16:21                        | 
| 8x H100s | OSFT (No Liger)           | Granite 4   | Table-GPT | 00:10:39                        | 
| 8x H100s | OSFT (No Liger)           | Granite 4   | Bespoke   | 00:16:46                        | 
| 8x H100s | Adjusted OSFT             | Granite 3.3 | Table-GPT | **00:44:25**                    | 
| 8x H100s | Adjusted OSFT             | Granite 3.3 | Bespoke   | **05:10:34**                    | 
| 8x H100s | Adjusted OSFT (No Liger)  | Granite 3.3 | Table-GPT | **00:45:23**                    | 
| 8x H100s | Adjusted OSFT (No Liger)  | Granite 3.3 | Bespoke   | **05:16:05**                    | 
| 8x H100s | Adjusted OSFT (No Liger)  | Granite 4   | Table-GPT | **00:07:26**                    | 
| 8x H100s | Adjusted OSFT (No Liger)  | Granite 4   | Bespoke   | **00:36:16**                    | 

#### Standard OSFT results (max_seq_len=4096, max_tokens_per_gpu=10000) vs Adjusted OSFT results (max_seq_len=20000, max_tokens_per_gpu=25000) 

| Hardware | Training Type    | Model       | Dataset   | Time per Epoch no Adjustment  | Time per Epoch with Adjustment | Difference in Time | Multiplier |
| ---------| ---------------- | ----------- | --------- | ----------------------------- | ------------------------------ | ------------------ | ---------- |
| 8x A100s | OSFT             | Granite 3.3 | Table-GPT | 00:36:09                      | 00:34:23                       | 00:01:46           | 0.95x      |
| 8x A100s | OSFT             | Granite 3.3 | Bespoke   | 00:58:39                      | 04:01:59                       | 03:03:20           | 4.13x      |
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | Table-GPT | 00:37:32                      | 00:35:42                       | 00:01:50           | 0.95x      |
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | Bespoke   | 01:00:48                      | 04:11:20                       | 03:10:32           | 4.13x      |
| 8x A100s | OSFT (No Liger)  | Granite 4   | Table-GPT | 00:14:11                      | 00:09:06                       | 00:05:05           | 0.64x      |
| 8x A100s | OSFT (No Liger)  | Granite 4   | Bespoke   | 00:23:01                      | 00:51:48                       | 00:28:47           | 2.25x      |
| 8x H100s | OSFT             | Granite 3.3 | Table-GPT | 00:46:04                      | 00:44:25                       | 00:01:39           | 0.96x      |
| 8x H100s | OSFT             | Granite 3.3 | Bespoke   | 01:15:08                      | 05:10:34                       | 03:55:26           | 4.13x      |
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Table-GPT | 00:46:51                      | 00:45:23                       | 00:01:28           | 0.97x      |
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Bespoke   | 01:16:21                      | 05:16:05                       | 03:59:44           | 4.14x      |
| 8x H100s | OSFT (No Liger)  | Granite 4   | Table-GPT | 00:10:39                      | 00:07:26                       | 00:03:13           | 0.70x      |
| 8x H100s | OSFT (No Liger)  | Granite 4   | Bespoke   | 00:16:46                      | 00:36:16                       | 00:19:30           | 2.16x      |

#### SFT vs Adjusted OSFT (No Liger)

| Hardware | Model       | Dataset   | SFT Time per Epoch | Adjusted OSFT (No Liger) Time per Epoch | Difference in Time | Multiplier |
| ---------| ----------- | --------- | ------------------ | --------------------------------------- | ------------------ | ---------- |
| 8x A100s | Granite 3.3 | Table-GPT | 00:10:01           | 00:35:42                                | 00:25:41           | 3.56x      |      
| 8x A100s | Granite 3.3 | Bespoke   | 01:17:02           | 04:11:20                                | 02:54:18           | 3.26x      |
| 8x A100s | Granite 4   | Table-GPT | 00:07:35           | 00:09:06                                | 00:01:31           | 1.2x       |
| 8x A100s | Granite 4   | Bespoke   | 00:42:48           | 00:51:48                                | 00:09:00           | 1.21x      |
| 8x H100s | Granite 3.3 | Table-GPT | 00:04:35           | 00:45:23                                | 00:40:48           | 9.90x      |
| 8x H100s | Granite 3.3 | Bespoke   | 00:35:47           | 05:16:05                                | 04:40:18           | 8.83x      | 
| 8x H100s | Granite 4   | Table-GPT | 00:05:40           | 00:07:26                                | 00:01:46           | 1.31x      |
| 8x H100s | Granite 4   | Bespoke   | 00:26:19           | 00:36:16                                | 00:09:57           | 1.38x      |


## Appendix: Full Training Results

The following table shows some examples of the amount of time needed to train the model for all three epochs (the default in `sft_granite_example.py` and `osft_granite_example.py`). Only the time spent actually training is counted; for example, the amount of time needed to load and save the model is not included in these measurements. This table is meant to be less comprehensive, and these experiments were done to ensure that the per-epoch results would provide a reasonable estimate while costing less resources to compute. They are provided in case you find them useful for seeing how the first epoch measurements compare to the full training cycle measures.

**All of the measured times are for a single trial only! They are NOT the average of multiple trials**

| Hardware | Training Type    | Model       | Dataset   | Full Training Time (HH:MM:SS) |
| ---------| ---------------- | ----------- | --------- | ----------------------------- |
| 8x A100s | SFT              | Granite 3.3 | Table-GPT | 00:32:42                      | 
| 8x A100s | SFT              | Granite 3.3 | Bespoke   | 03:57:34                      | 
| 8x A100s | OSFT             | Granite 3.3 | Table-GPT | 02:00:52                      | 
| 8x A100s | OSFT             | Granite 3.3 | Bespoke   | 03:09:33                      | 
