#!/bin/bash
#SBATCH -J debate_youcook
#SBATCH -o /public/home/dzhang/pyProject/hytian/XModel/Multi-Agents-Debate-main/logs/debate_youcook.log
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:2
#SBATCH -t 3-00:00:00

source /public/home/dzhang/anaconda3/bin/activate
conda activate qwen2

CUDA_VISIBLE_DEVICES=0,1 python code/debate4cp_lora_r1_llama2_dialogue_lastest_intern8.py \
    -o data/youcook_test/output_r1_llama2x_videollava_ep5_interv5_coco1 \


