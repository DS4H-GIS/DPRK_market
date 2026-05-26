# North Korea Market Detection

## Introduction
This repository contains the official implementation of  
**"Measuring Market Activity with AI and High-Resolution Satellite Imagery: Evidence from North Korea"**

The project leverages a deep learning–based segmentation model to detect North Korean markets from satellite imagery.

---

## Project Structure
```
DPRK_market/
├── model/                    # Model definition files
├── multiyear/
│   ├── NK_market_2_typeA/
│   │   ├── image/            # Satellite images (not uploaded due to copyright)
│   │   └── label/            # Binary segmentation labels
│   └── metadata/
│       └── NK_market_v2_typeA/
│           ├── NK_market_v2_typeA_train_metadata.csv
│           └── NK_market_v2_typeA_test_metadata.csv
├── result/                   # Inference output
├── augmentation.py           # Data augmentation functions
├── dataloader.py             # Data loading scripts
├── eval.py                   # Evaluation / inference script
├── main.py                   # Main training entry point
└── utils.py                  # Utility functions
```

## Installation
```bash
git clone https://github.com/DS4H-GIS/DPRK_market.git
cd DPRK_market
```

## Train
```bash
python main.py --epoch 100 --lr 0.01 --weight 0.4
```
**Arguments**
- `--epoch`: number of training epochs
- `--weight`: class weight for the non-market category (used in Focal Loss)
- `--lr`: learning rate

## Checkpoint
You can download our pretrained model [here](https://drive.google.com/file/d/1f3eGfvlVkBiHgicOJN6wougZug10DrT3/view?usp=drive_link)

## Evaluation
```bash
python eval.py --model <path_to_model_checkpoint>
```
**Arguments**
- `--model`: path to the trained model checkpoint for evaluation

## Requirements
- Python >= 3.7
- PyTorch >= 1.8.1
- CUDA >= 11.1
