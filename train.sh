#!/bin/bash

# setting up default hyperparameters
seed=42
subsample_ratio=1.0 # change this parameter to run the scaling plot
model_name="meta-llama/Meta-Llama-3-8B"
split=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) seed="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --rr) rr="$2"; shift 2;;
        --epochs) epochs="$2"; shift 2 ;;
        --bs) bs="$2"; shift 2 ;;
        --wd) wd="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --task_name) task_name="$2"; shift 2 ;;
        --split) split="$2"; shift 2 ;;
        --subsample_ratio) subsample_ratio="$2"; shift 2 ;;
        --model_name) model_name="$2"; shift 2 ;;
        --rh) rh=true; shift ;;
        --exp_id) exp_id="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

per_device_train_batch_size=2
# Get the number of GPUs + 1
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$((bs / (gpu_count * per_device_train_batch_size)))
echo "Running experiment with gpu_count $gpu_count, per_device_train_batch_size $per_device_train_batch_size, grad_acc $grad_acc, and thus effective bs $bs"

pretty_name=${model_name##*/}
pretty_name=$(echo "$pretty_name" | sed 's/-//g')
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-${split}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
else
    run_name="${task_name}-${split}-subsample-${subsample_ratio}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
echo "Running experiment with run name: $run_name"
root_dir=$([ "$rh" = true ] && echo "/new_data/wenlong_rh" || echo "/new_data/wenlong")
output_dir="${root_dir}/${exp_id}/${run_name}"

# Execute the training command with the specific hyperparameters
torchrun --nproc_per_node=$gpu_count  train.py \
    --seed=$seed \
    --model_name=$model_name \
    --block_size=2048 \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --per_device_eval_batch_size=3 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --rehersal_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --task_name=$task_name \
    --split=$split \
    --logging_steps=10 \
    --run_name=$run_name \
    --bf16=True \
    --output_dir=$output_dir \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --report_to="tensorboard" \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --log_level="info" \
    --fsdp="hybrid_shard auto_wrap" \
    --fsdp_config="training_scripts/config/fsdp_config.json"

#     --evaluation_strategy="no"
# /new_data/knowledge_rh/quality/synth_knowledge2_0/training_mix/combined_cut_100x_cpt_non_tokenized.jsonl  existing data

# to run it we do:
# command: ./training_scripts/train.sh \
# 	--lr 5e-06 \
# 	--rr 0.0 \
# 	--epochs 2 \
# 	--bs 16 \
# 	--wd 0.01 \
# 	--warmup 0.05 \
# 	--task_name quality \
# 	--split synth_knowledge2_0_xcombined_cut_100x_cpt_non_tokenized \
# 	--rh \
# 	--model_name meta-llama/Llama-3.1-8B-Instruct \
# 	--exp_id llama_3_1_8b_instruct_combined_cut_50x_key_facts_inst
