# IMAGGarment-1
## How to test
```
python inference_logo.py \
--stage1_model_ckpt [stage1 checkpoint] \
--stage2_model_ckpt [stage2 chekcpoint] \
--sketch_path [your sketch path] \
--logo_path [your logo path] \
--mask_path [your mask path] \
--color_path [your color path] \
--prompt [your prompt] \
--output_path [your save path] \
--ip_ckpt [color adapter checkpoint] \
--device [your device]
```