CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "/home/yyshi/openvla-7b" \
  --data_root_dir /home/yyshi/bridge_data  \
  --dataset_name bridge_orig \
  --run_root_dir /home/yyshi/experiments \
  --adapter_tmp_dir /home/yyshi/experiments_tmp \
  --max_steps 200 \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project vla \
  --wandb_entity vla \
  --save_steps 200
