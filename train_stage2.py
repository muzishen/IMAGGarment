import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import logging
import torch

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
from adapter.utils import is_torch2_available
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)
if is_torch2_available():
    from adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor ,SkipAttnProcessor
else:
    from adapter.attention_processor import IPAttnProcessor, AttnProcessor



# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, args,size=512,  i_drop_rate=0.1, image_root_path=""):
        super().__init__()

        self.size = size
        self.i_drop_rate = i_drop_rate
        self.image_root_path = image_root_path
        self.args = args
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8) 
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True) 
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        logo = item["logo"]
        cloth = item['sketch']
        mask = item['mask']
        gt = item['cloth']

        logo = Image.open(os.path.join(self.image_root_path, logo))  
        cloth=Image.open(os.path.join(self.image_root_path, cloth))
        mask = Image.open(os.path.join(self.image_root_path, mask))
        gt = Image.open(os.path.join(self.image_root_path,gt))
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        else:
            drop_image_embed = 0
    
        
        return {
            "cloth": self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            "logo" : self.vae_processor.preprocess(logo, self.args.height, self.args.width)[0],
            "mask" : self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0],
            'gt'   : self.vae_processor.preprocess(gt, self.args.height, self.args.width)[0],
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    clothes = torch.stack([example["cloth"] for example in data])
    logos = torch.stack([example["logo"] for example in data])
    masks = torch.stack([example["mask"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    gts = torch.stack([example["gt"] for example in data])
    return {
        "cloth": clothes,
        "logo" : logos,
        "mask" : masks,
        "gt"   : gts,
        "drop_image_embeds": drop_image_embeds
    }
    

class Stage2(torch.nn.Module):
    def __init__(self, unet,attn_block, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.attn_block = attn_block

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps):
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=None).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

 
def check_inputs( image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask 
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        else:
            attn_procs[name] = SkipAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(attn_procs)
    
    for name, param in unet.named_parameters():
            if 'attn1' in name:
                param.requires_grad_(True)
    attn_blocks = torch.nn.ModuleList()
    
    for name, param in unet.named_modules():
        if "attn1" in name:
            attn_blocks.append(param)
            
    if args.resume_from_checkpoint is not None:        
            sd_state = torch.load(args.resume_from_checkpoint, map_location="cpu",weights_only=True)
            attn_sd = {}
            for k in sd_state:
                if 'attn_block' in k:
                    attn_sd[k.replace('attn_block.','')] = sd_state[k]
            attn_blocks.load_state_dict(attn_sd)
   
    stage2 = Stage2(unet,attn_blocks)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(stage2.attn_block.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, args, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    stage2, optimizer, train_dataloader = accelerator.prepare(stage2, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(stage2):
                
                #prepar image
                concat_dim = -2  # FIXME: y axis concat
                image , condition_image , mask = batch['cloth'],batch['logo'],batch['mask']
                image, condition_image, mask = check_inputs(image, condition_image, mask,args.width,args.height)
                image = prepare_image(image).to(accelerator.device, dtype=weight_dtype)
                condition_image = prepare_image(condition_image).to(accelerator.device, dtype=weight_dtype)
                mask = prepare_mask_image(mask).to(accelerator.device, dtype=weight_dtype)
                # Mask image
                masked_image = image * (mask < 0.5)
                # VAE encoding
                image_latent = compute_vae_encodings(image,vae)
                masked_latent = compute_vae_encodings(masked_image, vae)
                condition_latent = compute_vae_encodings(condition_image, vae)
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(condition_latent, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
                condition_latent = image_embeds
                
                mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
                # Concatenate latents
                masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
                mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
                
                
                gt = batch['gt']
                gt = prepare_image(gt).to(accelerator.device, dtype=weight_dtype)
                gt_latent = compute_vae_encodings(gt,vae)
                image_latent_concat = torch.cat([gt_latent,condition_latent],dim=concat_dim)
                # image_latent_concat = torch.cat([image_latent,condition_latent],dim=concat_dim)
                del image, mask, condition_image
                

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(masked_latent_concat)
                bsz = masked_latent_concat.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=masked_latent_concat.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(image_latent_concat, noise, timesteps)
                
                #concat  in channel
                inpainting_latent_model_input = torch.cat([noisy_latents, mask_latent_concat, masked_latent_concat], dim=1)
                noise_pred = stage2(inpainting_latent_model_input, timesteps)
        
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path,safe_serialization=False)
                print(f"Success save checkpoint-{global_step}")
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
    