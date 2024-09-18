# PiLaMIM: Toward Richer Visual Representation by Integrating Pixel and Latent Masked Image Modeling

## Setting
conda create -n name python=3.8
pip install -r requirements.txt

## Pretrain
```bash
python pretrain.py
```

## Linear probing
```bash
python linear_probing.py --dataset data_name --pretrained_model_path model_path
```

All datasets were utilized with torchvision
data_name : imagenet1k, cifar100, cifar10, Places365, INat2021, CLEVR, CLEVR_Dist