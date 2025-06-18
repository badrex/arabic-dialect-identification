
# Arabic Dialect Identification (ADI) in Speech by Fine-tuning Speech Encoders


<img src="https://huggingface.co/badrex/mms-300m-arabic-dialect-identifier/resolve/main/assets/logo_2.png" alt="Image header" width="500"/>

https://huggingface.co/badrex/mms-300m-arabic-dialect-identifier/resolve/main/assets/logo_2.png

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/badrex/mms-300m-arabic-dialect-identifier)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2505.24713)
[![Language](https://img.shields.io/badge/Language-Arabic-red)](https://en.wikipedia.org/wiki/Arabic)



Hello there 👋🏼

This is the code repo for our Interspeech 2025 paper **Voice Conversion Improves Cross-Domain Robustness for Spoken Arabic Dialect Identification**. 



## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch
- Transformers
- Datasets
- Evaluate
- Check out requirement.txt

### Environment Setup

1. Set up your environment variables in a `.env` file:
```
WANDB_API_KEY=your_wandb_key
HF_API_KEY=your_huggingface_key
HF_HOME=/path/to/huggingface/cache
```

2. Make sure you have access to the Common Voice dataset on Hugging Face.

# Training a Model


1. install requirements
```
> pip install -r requirements.txt
```

2. Make sure you have the data as Hugging Face datasets and have paths in the config.yaml file

```
project: adi5-vc

# choose a multilingual pre-trained speech encoder
pretrained_model:  facebook/mms-300m  # or "facebook/wav2vec2-xls-r-300m"

# datasets settings
natural_dataset: adi5-aljazeera-arabic-dialects # path to natural speech dataset
resynthesized_dataset: adi5-aljazeera-arabic-dialects-vc-12-spkrs-cv # path to resynthesized dataset using vc or augmentation
sample_data: false 
target_speakers:  null # or [1, 3, 4, 6], choose from 1-6 spaker ID 
sample_size: 14589 # 14589 means no sampling

# training settings
learning_rate: 0.00005  
batch_size: 16
num_train_epochs: 3
apply_vc: true # whether or not to use voice conversion as data augmentation
add_natural_data: true
max_duration: 10 # shorter audio duration (=10 sec) is better for training, faster to train and higher accuracy
apply_dropout: false
freeze_feature_extractor: true
random_seed: 842

```


3. Run the scripts 
```
> python3 train_ADI5_with_vc.py --config config.yaml
```

# Citation
If you use this code or our ADI model, please cite our paper as 


```
@inproceedings{abdullah2025voice,
  title={Voice Conversion Improves Cross-Domain Robustness for Spoken Arabic Dialect Identification},
  author={Badr M. Abdullah and Matthew Baas and Bernd Möbius and Dietrich Klakow},
  year={2025},
  publisher={Interspeech},
  url={https://arxiv.org/pdf/2505.24713}
}

```




