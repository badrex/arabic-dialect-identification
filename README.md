
### How to train the model

1. install requirements
```
> pip install -r requirements.txt
```

2. Make sure you have the data as Hugging Face datasets and have paths in the config.yaml file

```
project: adi5-vc
pretrained_model:  facebook/mms-300m  # or , "facebook/wav2vec2-xls-r-300m", facebook/wav2vec2-large-xlsr-53
natural_dataset: adi5-aljazeera-arabic-dialects # path to natural speech dataset
resynthesized_dataset: adi5-aljazeera-arabic-dialects-vc-12-spkrs-cv # path to resynthesized dataset using vc or augmentation
sample_data: false #true
target_speakers:  null # or [1, 3, 4, 6], choose from 1-6 spaker ID 
sample_size: 14589 # 14589 means no sampling
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

3. You need your weights & biases and hugging face access tokes to be stored in an .env file

4. Run the scripts 
```
python3 train_ADI5_with_vc.py --config config.yaml
```


