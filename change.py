import torch
import os
save_folder='save_path'
ckpt = "/path_to/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu",weights_only=True)
image_proj_sd = {}
ip_sd = {}
for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

torch.save({"image_proj": image_proj_sd, "color_adapter": ip_sd}, os.path.join(save_folder,"color_adapter.bin"))