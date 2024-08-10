# ECG_SSL_LFBT

# Introduction
Code implementation of **Lead-Fusion Barlow Twins (LFBT)**, a new self-supervised learning (SSL) method for multilead electrocardiograms (ECGs).
# Requirements
```
numpy==1.22.4
scikit_learn==1.3.0
scipy==1.10.1
torch==1.11.0
torchvision==0.12.0
tqdm==4.61.2
```
# Data
- The Ningbo First Hospital (NFH), PTB-XL, CPSC, and Chapman database can be downloaded from [https://physionet.org/content/challenge-2021/1.0.3/](https://physionet.org/content/challenge-2021/1.0.3/)
- The Shanghai Ninth People's Hospital (SNPH) database is confidential. 
- Each ECG should be transferred to a numpy array (shape: C(lead)×L(length)) and saved as an npy file (.npy). Z-score normalizes each ECG.
- The dataset for pretraining should be structured as follows:
```bash
pt_data_dir/
├── samples
    ├── sample_0.npy
    ├── sample_1.npy
    └── ...
```

- The dataset for a downstream task should be structured as follows:
```bash
down_data_dir/
├── class_x
│   ├── sample_0.npy
│   ├── sample_1.npy
│   └── ...
└── class_y
    ├── sample_0.ext
    ├── sample_1.ext
    └── ...
```