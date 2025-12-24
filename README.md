# MsAutoformer: Multi-Scale Autoformer for Reservoir Fluid Identification

# Overview
This project implements **MsAutoformer**, an improved Autoformer architecture for reservoir fluid identification in geoscience applications.


## Project Structure

```
├── model_MsAutoformer.py      # Core model implementation
├── layers/
│   ├── AutoCorrelation.py     # AutoCorrelation mechanism
│   ├── Autoformer_EncDec.py   # Encoder layers
│   └── Embed.py               # Embedding layers
├── train.py                   # Training script
├── test.py                    # Testing and evaluation
├── predict_for_project.py     # Inference script
├── tool_for_pre.py            # Data preprocessing
├── tool_for_train.py          # Training utilities
├── tool_for_test.py           # Testing utilities
└── data_pre.py                # Data preparation
```

## Usage

### 1. Training
```bash
python train.py
```


## Requirements
- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

