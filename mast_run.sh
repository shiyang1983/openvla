TORCHXCONFIG=~/fbsource/fbcode/scripts/yyshi/.torchxconfig_metaconda \
  torchx run \
  --scheduler_args="forceSingleRegion=False;fbpkg_ids=torchx_conda_mount:stable,manifold.manifoldfs:prod,oil.oilfs:stable"  \
  fb.conda.torchrun \
  --h grandteton \
  --run_as_root True \
  --env "DISABLE_OILFS=1;MANIFUSE_BUCKET=xr_core_ai_asl_llm;LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so" \
  --name "vla" \
  -- \
  --no-python --nnodes=1 --nproc-per-node=8 \
  ./run.sh  vla-scripts/finetune.py \
  --vla_path /mnt/xr_core_ai_asl_llm/tree/vla/models/openvla-7b \
  --data_root_dir /mnt/xr_core_ai_asl_llm/tree/vla/data/bridge_data  \
  --dataset_name bridge_orig \
  --run_root_dir /mnt/xr_core_ai_asl_llm/tree/vla/experiments/03172025_1127 \
  --adapter_tmp_dir /mnt/xr_core_ai_asl_llm/tree/vla/experiments/03172025_1127_tmp \
  --max_steps 200 \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project vla \
  --wandb_entity vla \
  --save_steps 200
