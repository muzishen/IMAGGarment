# <img src="./assets/logo.png" alt="Logo" width="30" height="30"/>IMAGGarment: Fine-Grained Garment Generation for Controllable Fashion Design



<a href='https://revive234.github.io/imaggarment.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2504.13176'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/keimen/IMAGGarment'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/datasets/IMAGDressing/IGPair'><img src='https://img.shields.io/badge/Dataset-GarmentBench-orange'></a>
[![GitHub stars](https://img.shields.io/github/stars/muzishen/IMAGGarment-1?style=social)](https://github.com/muzishen/IMAGGarment-1)


## 🗓️ Release

- [2025/4/18] 🔥 We released the [technical report](https://arxiv.org/pdf/2504.13176) of IMAGGarment-1.
- [2025/4/18] 🔥 We release the train and inference code of IMAGGarment-1.
- [2025/4/17] 🎉 We launch the [project page](https://revive234.github.io/imaggarment.github.io/) of IMAGGarment-1.






## 💡 Introduction
IMAGGarment-1 addresses the challenges of multi-conditional controllability in personalized fashion design and digital apparel applications.
Specifically, IMAGGarment-1 employs a two-stage training strategy to separately model global appearance and local details, while enabling unified and controllable generation through end-to-end inference.
In the first stage, we propose a global appearance model that jointly encodes silhouette and color using a mixed attention module and a color adapter.
In the second stage, we present a local enhancement model with an adaptive appearance-aware module to inject user-defined logos and spatial constraints, enabling accurate placement and visual consistency.
![architecture](./assets/architecture.jpg)

## 🚀 Dataset Demo

![dataset_demo](./assets/dataset_sample.jpg)
## 🚀 Examples

![results_1](./assets/introduction.jpg)


### Different Colors and Silhouettes
<p align="center">
  <img src="./assets/app1-1.png" alt="results_2" width="800" height="400"/>
</p>

<p align="center">
  <img src="./assets/app1-2.png" alt="results_2" width="800" height="400"/>
</p>


### More Results
<p align="center">
  <img src="./assets/appendix1-1.png" alt="results_2" width="800" height="400"/>
</p>

<p align="center">
  <img src="./assets/appendix1-2.png" alt="results_2" width="800" height="400"/>
</p>


## 🔧 Requirements

- Python>=3.8
- [PyTorch>=2.0.0](https://pytorch.org/)
- cuda>=11.8
```
conda create --name IMAGGarment python=3.8.8
conda activate IMAGGarment
pip install -U pip

# Install requirements
pip install -r requirements.txt
```
## 🌐 Download Models

You can download our models from [百度云](https://pan.baidu.com/s/1_nlOTiLqeRYBNb0w0249vQ?pwd=ky3q#list/path=%2F). You can download the other component models from the original repository, as follows.
- stabilityai/sd-vae-ft-mse(https://huggingface.co/stabilityai/sd-vae-ft-mse)
- if train: [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), if test: [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE)
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)

## 🚀 How to train
```
# Please download the GarmentBench data first 
# and modify the path in train_color_adapter.sh, train_stage1.sh and train_stage2.sh

# train color adapter
sh train_color_adapter.sh
# Once training of color adapter is complete, you can convert the weights into the desired format.
python change.py

# train GAM model
sh train_GAM.sh
# train LEM model
sh train_LEM.sh
```
## 🚀 How to test
```
python inference_IMAGGarment-1.py \
--GAM_model_ckpt [GAM checkpoint] \
--LEM_model_ckpt [LEM chekcpoint] \
--sketch_path [your sketch path] \
--logo_path [your logo path] \
--mask_path [your mask path] \
--color_path [your color path] \
--prompt [your prompt] \
--output_path [your save path] \
--color_ckpt [color adapter checkpoint] \
--device [your device]
```
## Acknowledgement
We would like to thank the contributors to the [IMAGDressing](https://github.com/muzishen/IMAGDressing) and [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) repositories, for their open research and exploration.

The IMAGGarment code is available for both academic and commercial use. Users are permitted to generate images using this tool, provided they comply with local laws and exercise responsible use. The developers disclaim all liability for any misuse or unlawful activity by users.
## Citation
If you find IMAGDressing-v1 useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{shen2025imaggarment,
  title={IMAGGarment-1: Fine-Grained Garment Generation for Controllable Fashion Design},
  author={Shen, Fei and Yu, Jian and Wang, Cong and Jiang, Xin and Du, Xiaoyu and Tang, Jinhui},
  booktitle={arXiv preprint arXiv:2504.13176},
  year={2025}
}
```
## 🕒 TODO List
- [x] Paper
- [x] Train Code
- [x] Inference Code
- [ ] GarmentBench Dataset
- [x] Model Weights
- [ ] Upgraded Version for High-resolution Images

## 👉 **Our other projects:**  
- [IMAGEdit](https://github.com/XWH-A/IMAGEdit): Training-Free Controllable Video Editing with Consistent Object Layout.  [可控多目标视频编辑]
- [IMAGDressing](https://github.com/muzishen/IMAGDressing): Controllable dressing generation. [可控穿衣生成]
- [IMAGGarment](https://github.com/muzishen/IMAGGarment): Fine-grained controllable garment generation.  [可控服装生成]
- [IMAGHarmony](https://github.com/muzishen/IMAGHarmony): Controllable image editing with consistent object layout.  [可控多目标图像编辑]
- [IMAGPose](https://github.com/muzishen/IMAGPose): Pose-guided person generation with high fidelity.  [可控多模式人物生成]
- [RCDMs](https://github.com/muzishen/RCDMs): Rich-contextual conditional diffusion for story visualization.  [可控故事生成]
- [PCDMs](https://github.com/tencent-ailab/PCDMs): Progressive conditional diffusion for pose-guided image synthesis. [可控人物生成]
- [V-Express](https://github.com/tencent-ailab/V-Express/): Explores strong and weak conditional relationships for portrait video generation. [可控数字人生成]
- [FaceShot](https://github.com/open-mmlab/FaceShot/): Talkingface plugin for any character. [可控动漫数字人生成]
- [CharacterShot](https://github.com/Jeoyal/CharacterShot): Controllable and consistent 4D character animation framework. [可控4D角色生成]
- [StyleTailor](https://github.com/mahb-THU/StyleTailor): An Agent for personalized fashion styling. [个性化时尚Agent]
- [SignVip](https://github.com/umnooob/signvip/): Controllable sign language video generation. [可控手语生成]

## 📨 Contact
If you have any questions, please feel free to contact with us at shenfei140721@126.com and jianyu@njust.edu.cn.
