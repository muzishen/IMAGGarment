  CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3  --mixed_precision "fp16"  --num_machines 1  --dynamo_backend "no" --main_process_port 29501\
  train_stage2.py \
  --pretrained_model_name_or_path="/path_to/stable-diffusion-inpainting" \
  --data_json_file="/path_to/GarmentBench/logo_pair.json" \
  --data_root_path="/path_to/GarmentBench" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=26 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/save_path" \
  --save_steps=1000  \
  --num_train_epochs 1000 


