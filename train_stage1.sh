export HOST_NUM=1
accelerate launch --gpu_ids 0,1,2,3 --use_deepspeed --num_processes 4  --main_process_port 12012\
  --deepspeed_config_file zero_stage2_config.json \
  train_stage1.py \
  --pretrained_model_name_or_path="/path_to/stable-diffusion-v1-5" \
  --pretrained_vae_model_path="/path_to/sd-vae-ft-mse" \
  --dataset_json_path="/path_to/GarmentBench/sketch_pair.json" \
  --clip_penultimate=False \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000 \
  --learning_rate=1e-5 \
  --weight_decay=0.01 \
  --lr_scheduler="constant" --num_warmup_steps=2000 \
  --output_dir="/save_path" \
  --checkpointing_steps=1000 
