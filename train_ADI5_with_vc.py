#!/usr/bin/env python
# scripts/train.py

import os
import sys
from pathlib import Path
import argparse
import time
import logging
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import Counter


import pandas as pd
import numpy as np
import torch
import wandb

# A function to load environment variables from .env file
def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key] = value.strip("'").strip('"')
                    except ValueError:
                        # Skip lines that don't have the format KEY=VALUE
                        continue
        return True
    return False

# Try to load from .env file in project root
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
env_path = project_root / '.env'
load_env_file(env_path)

# Add project root to Python path
sys.path.insert(0, str(project_root))

# import Hugging Face libraries
import evaluate
from datasets import (
    load_dataset, 
    load_from_disk, 
    DatasetDict, 
    concatenate_datasets, 
    Audio
)
from transformers import (
    AutoModelForAudioClassification, 
    AutoFeatureExtractor, 
    Wav2Vec2Config,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)

from huggingface_hub import login

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("train_ADI5_model")

# a function to load input config parameters
def load_config_parameters(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load hyperparameters from the YAML file
# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Read hyperparameters from a YAML file.'
)

parser.add_argument(
    '--config', 
    type=str, 
    required=True, 
    help='Path to the YAML configuration file.'
)

args = parser.parse_args()

logger.info("Loading configuration parameters...")
config_parameters = load_config_parameters(args.config)

# check out the config parameters
logger.info(config_parameters)

# Set the seed for reproducibility
logger.info("Setting random seed:")
set_seed(config_parameters['random_seed'])

# login to wandb
logger.info("Logging in to Weights & Biases (wandb)...")
wandb_key = os.environ.get("WANDB_API_KEY")

if not wandb_key:
    raise ValueError("WANDB_API_KEY environment variable is not set."
                     f" Please set it in your .env file.")

# start a new wandb run to track this script
wandb.init(
    # token is automatically set from the environment variable
    # set the wandb project where this run will be logged
    project=config_parameters['project'],

    # track hyperparameters and run metadata
    config={
        'learning_rate': config_parameters['project'],
        'sample_size': config_parameters['sample_size'],
        'random_seed': config_parameters['random_seed'],
        'apply_vc': 'vc' if config_parameters['apply_vc'] else 'n',
    }
)

# login to HF hub
logger.info("Logging in to Hugging Face Hub...")
login(os.environ.get("HF_API_KEY"))

if not os.environ.get("HF_API_KEY"):
    raise ValueError("HF_API_KEY environment variable is not set."
                     " Please set it in your .env file.")

# make labels more human-readable
dir_to_dialect = {
    'EGY': 'Egyptian',
    'GLF': 'Gulf',
    'LAV': 'Levantine',
    'MSA': 'MSA',
    'NOR': 'Maghrebi',

    # add identity mapping for full name dialects
    'Egyptian': 'Egyptian',
    'Gulf': 'Gulf',
    'Levantine': 'Levantine',
    'Maghrebi': 'Maghrebi',
}

# load the pretrained model and feature extractor
logger.info("Specifying pretrained model and feature extractor...")
model_id = config_parameters['pretrained_model']

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, 
    do_normalize=True,
    return_attention_mask=True
)

# laod natural dataset from disk for dev set
logger.info("Loading natural speech ADI5 dataset from disk...")

# I was thinking of not loading the natural speech dataset since we don't 
# use in all setups, but then I realized that we need it for dev set 
# and also for obtaining the label mapping
ADI5_dataset_natural = load_from_disk(config_parameters['natural_dataset'])
ADI5_dataset_dev = ADI5_dataset_natural['dev']

# change the dialect name according to the mapping
logger.info("Mapping dialect names in dev set to human-readable format...")
ADI5_dataset_dev = ADI5_dataset_dev.map(
    lambda x: {'dialect': dir_to_dialect[x['dialect']]},
)

logger.info("Dev set structure and dialects:")
logger.info(ADI5_dataset_dev)
logger.info(ADI5_dataset_dev.unique('dialect'))

# sample N instances from training data
# ADI5_dataset_train = ADI5_dataset['train'].shuffle().select(
#     range(config_parameters['sample_size'])
# )

#print(ADI5_dataset_train)

# if vc is enabled, use the vc dataset as the training dataset
if config_parameters['resynthesized_dataset'] != "all-augmented-datasets":
    vc_dataset_name = config_parameters['resynthesized_dataset']

else:
    # make a list of augmented dataset names 
    augmented_datasets_names = [
        "adi5-aljazeera-arabic-dialects-additive-noise", 
        "adi5-aljazeera-arabic-dialects-pitch-shift", 
        "adi5-aljazeera-arabic-dialects-spec-augment",
        "adi5-aljazeera-arabic-dialects-RIR",
    ]

logger.info(config_parameters['resynthesized_dataset'])

# WARNING: This part of the code is not well-structured and needs refactoring
# TODO: Refactor this part to make it more readable and maintainable
# ----------------------------------------------------------------------------
# BEGINNING OF THE MESSY PART
# ----------------------------------------------------------------------------

# if natural speech is used for the experiment
if not config_parameters['apply_vc'] or config_parameters['add_natural_data']:
    # shuffle the natural dataset and select samples


    # comment shuffling for now
    logger.info("shuffling natural dataset for training...")
    ADI5_dataset_natural_train = ADI5_dataset_natural['train'] #.shuffle(
    #     seed=config_parameters['random_seed'],
    #     num_proc=10,  # number of processes to use for shuffling
    # )
    
    # sampling is commented out for now
    # TODO: figure out a better way to structure this 
    # ADI5_dataset_natural_train = ADI5_dataset_natural_train.select(
    #     range(config_parameters['sample_size'])
    # )

    if config_parameters['sample_data']:
        ADI5_dataset_natural_train = ADI5_dataset_natural_train.select(
            range(config_parameters['sample_size'])
        )

        # change the dialect name according to the mapping
        # ADI5_dataset_natural_train = ADI5_dataset_natural_train.map(
        #     lambda x: {'dialect': dir_to_dialect[x['dialect']]},
        # )

        segment_ids = ADI5_dataset_natural_train['segment_id']
        print("Natural dataset: ", ADI5_dataset_natural_train)


if config_parameters['apply_vc']:

    # if we use a combination of all augmented datasets
    if config_parameters['resynthesized_dataset'] == "all-augmented-datasets":

        # load the datasets and concatenate them

        print("Reading augmented datasets...")
        datasets = []

        for dataset_name in augmented_datasets_names:
            dataset = load_dataset(dataset_name)
            datasets.append(dataset['train'])

        # Combine all datasets into one
        combined_train = concatenate_datasets(datasets)
    
        ADI5_dataset_vc = DatasetDict({
            'train': combined_train
        })

    # otherwise, load the specified dataset
    else:
        ADI5_dataset_vc = load_from_disk(vc_dataset_name)


    # if we apply voice conversion, the select samples from target speakers 
    if config_parameters['target_speakers'] is not None:

        # comment shuffling for now
        ADI5_dataset_vc_train = ADI5_dataset_vc['train'] #.shuffle(
        #     seed=config_parameters['random_seed'],
        #     num_proc=10,  # number of processes to use for shuffling
        # )

        # get the speakers IDs
        ADI5_dataset_vc_train = ADI5_dataset_vc_train.filter(
            lambda x: x['speaker'] in config_parameters['target_speakers'], 
            num_proc=10,  # number of processes to use for filtering
        )
        
        # Create boolean masks for both speakers
        # speaker1_mask = [
        #     x['speaker'] == config_parameters['target_speakers'][0] 
        #     for x in ADI5_dataset_vc_train
        # ]

        # speaker2_mask = [
        #     x['speaker'] == config_parameters['target_speakers'][1] 
        #     for x in ADI5_dataset_vc_train
        # ]
        
        # # Get indices for both speakers
        # speaker1_indices = [i for i, mask in enumerate(speaker1_mask) if mask]
        # speaker2_indices = [i for i, mask in enumerate(speaker2_mask) if mask]
        
        # # take all the samples from the first speaker and the second speaker
        # final_indices = (
        #     speaker1_indices + speaker2_indices
        # )

        # Take only first 50% of speaker 1 and last 50% of speaker 2
        # final_indices = (
        #     speaker1_indices[:len(speaker1_indices)//2] + 
        #     speaker2_indices[len(speaker2_indices)//2:]
        # )
        
        # Create final dataset and shuffle
        # ADI5_dataset_vc_train = ADI5_dataset_vc_train.select(final_indices).shuffle(
        #     seed=config_parameters['random_seed']
        # )#.select(range(config_parameters['sample_size']))

        # add feature original_segment_id to the dataset
        logger.info("Adding original_segment_id to the dataset...")
        ADI5_dataset_vc_train = ADI5_dataset_vc_train.map(
            lambda x: {
                'original_segment_id': x['segment_id'].split('_')[0]
            },
            num_proc=10,  # number of processes to use for shuffling
        )

    # if using traditional data augmentation -- no voice conversion here
    elif config_parameters['target_speakers'] is None:
        # here just load the data augmented dataset and shuffle
        # comment shuffling for now
        ADI5_dataset_vc_train = ADI5_dataset_vc['train'] #.shuffle(
        #     seed=config_parameters['random_seed'],
        #     num_proc=10,  # number of processes to use for shuffling
        # )
        
        # ADI5_dataset_vc_train = ADI5_dataset_vc_train.select(
        #     range(config_parameters['sample_size'])
        # )

        # add feature original_segment_id to the dataset
        ADI5_dataset_vc_train = ADI5_dataset_vc_train.map(
            lambda x: {
                'original_segment_id': x['segment_id'].split('_')[0]
            },
            num_proc=10,  # number of processes to use for shuffling
        )

    # take only those samples that are present in the natural dataset
    # ADI5_dataset_vc_train = ADI5_dataset_vc_train.filter(
    #     lambda x: x['original_segment_id'] in segment_ids
    # )

    logger.info(f"VC dataset: {ADI5_dataset_vc_train}" )
    logger.info(f"Size of VC dataset: {len(ADI5_dataset_vc_train)}")
    

# train only on vc speech  
if config_parameters['apply_vc'] and not config_parameters['add_natural_data']:
    ADI5_dataset_train = ADI5_dataset_vc_train

    # show the dialects in the vc dataset
    logger.info(f"VC dataset dialects: {ADI5_dataset_train.unique('dialect')}")

# mix vc speech with natural speech
elif config_parameters['apply_vc'] and config_parameters['add_natural_data']:
    # concatenate natural and vc datasets
    ADI5_dataset_train = concatenate_datasets(
        [
            ADI5_dataset_natural_train, 
            ADI5_dataset_vc_train
        ]
    )
    
    logger.info("Shuffling overall training dataset...")
    ADI5_dataset_train = ADI5_dataset_train.shuffle(
        seed=config_parameters['random_seed'] #, num_proc=10
    )

    # show the dialects in the training dataset
    unique_dialects = ADI5_dataset_train.unique('dialect')
    logger.info(f"Training dataset dialects: {unique_dialects}")

# only natural speech
else:
    ADI5_dataset_train = ADI5_dataset_natural_train

# ----------------------------------------------------------------------------
# END OF THE MESSY PART 
# ----------------------------------------------------------------------------


logger.info("Dialect distribution in training set:")
logger.info(ADI5_dataset_train.unique('dialect'))
logger.info(Counter(ADI5_dataset_train['dialect']))

# change the dialect name according to the mapping
logger.info("Mapping dialect names in training set to human-readable format...")

def map_dialects(batch):
    return {
        'dialect': [dir_to_dialect[dialect] for dialect in batch['dialect']]
    }

ADI5_dataset_train = ADI5_dataset_train.map(
    map_dialects,
    batched=True,
    batch_size=1024,
    num_proc=1, # for batching this has to be 1
)

logger.info("Dialect distribution in training set after mapping:")
logger.info(ADI5_dataset_train.unique('dialect'))
logger.info(Counter(ADI5_dataset_train['dialect']))

# shuffle the training set
logger.info("Shuffling training dataset...")
ADI5_dataset_train = ADI5_dataset_train.shuffle(
    seed=config_parameters['random_seed'], 
    #num_proc=10
)

# for debugging purposes, take onyl the first 1000 samples
# logger.info("Taking first 100 samples from the training set for debugging...")
# ADI5_dataset_train = ADI5_dataset_train.select(
#     range(100)
# )

# to avoidd CUDA out of memory errors, clip audio samples to a maximum duration
logger.info("Clipping audio samples to a maximum duration...")

def clip_audio(example):
    """Clip audio samples to a maximum duration."""
    max_duration = config_parameters['max_duration']
    sampling_rate = feature_extractor.sampling_rate
    max_samples = int(max_duration * sampling_rate)
    
    # Clip the audio array
    if len(example['audio']['array']) > max_samples:
        example['audio']['array'] = example['audio']['array'][:max_samples]
    
    return example

# Use without batched=True (simpler and more reliable)
ADI5_dataset_train = ADI5_dataset_train.map(
    clip_audio,
    num_proc=4,  
)

ADI5_dataset_dev = ADI5_dataset_dev.map(
    clip_audio,
    num_proc=4,
)


logger.info(f"Final training dataset: {ADI5_dataset_train}")

# log the total number of samples in the training set
logger.info(f"Total samples in training set: {len(ADI5_dataset_train)}")

# create an index for each label 
logger.info("Creating label index mapping...")

str_to_int = {
    s: i for i, s in enumerate(ADI5_dataset_train.unique('dialect'))
}

logger.info(f"Label dict: {str_to_int}" )

# set max duration for audio samples
max_duration = config_parameters['max_duration']

# based on the model typel, set input features key
if model_id == "facebook/w2v-bert-2.0":
    input_features_key = "input_features"
else:
    input_features_key = "input_values"

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        return_attention_mask=True,
    )

    inputs["label"] = [str_to_int[x] for x in examples["dialect"]]
    # Ensure 'input_values' contains numerical arrays
    inputs[input_features_key] = [
        np.array(x) for x in inputs[input_features_key]
    ]

    return inputs

# obtain dev and test splits as dataset objects
#ADI5_dataset_dev = ADI5_dataset['dev']
#ADI5_dataset_test = ADI5_dataset['test']

# merge ADI5_dataset_dev and ADI5_dataset_test
# with ADI5_dataset_train_sample into single dataset
ADI5_sample = DatasetDict({
    'train': ADI5_dataset_train,
    'dev': ADI5_dataset_dev,
    #'test': ADI5_dataset_test
})

#logger.info("Training dataset with extracted features: ", ADI5_sample)

logger.info("Extract audio features using model front-end...")

ADI5_sample_encoded = ADI5_sample.map(
    preprocess_function,
    remove_columns=["audio"],
    batched=True,
    batch_size=64,
    num_proc=1, # for batching this has to be 1
)
logger.info(f"Training dataset with extracted features: {ADI5_sample_encoded}")

   
#true_labels_train_str = ADI5_sample_encoded["train"]['label']

int_to_str = {
    i: s for i, s in str_to_int.items()
}

num_labels = len(int_to_str)

logger.info(f"Integer to label dict: {int_to_str}")

# create a config object for the model
logger.info("Creating an instance of the pretraibed model...")
config = AutoConfig.from_pretrained(model_id)

config.num_labels=num_labels
config.label2id=str_to_int
config.id2label=int_to_str

# check if dropout is enabled
if config_parameters['apply_dropout']:
    config.hidden_dropout = 0.1           # Dropout for hidden states
    config.attention_dropout = 0.1        # Dropout in attention layers
    config.activation_dropout = 0.1       # Dropout after activation functions
    config.feat_proj_dropout = 0.1   


logger.info("Creating an instance of the pretrained model...")
model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    config=config,
)

# Freeze encoder
if (config_parameters['freeze_feature_extractor'] and 
    model_id != "facebook/w2v-bert-2.0"):
    logger.info("Freezing feature extractor parameters...")
    model.wav2vec2.feature_extractor._freeze_parameters()

# to confirm if feature extractor is frozen
# for name, param in model.named_parameters():
#     if "feature_extractor" in name:
#         print(f"{name}: {param.requires_grad}")


model_name = model_id.split("/")[-1]
batch_size = config_parameters['batch_size']
gradient_accumulation_steps = 1
num_train_epochs = config_parameters['num_train_epochs'] 

 # to make sure all models are trained for same number of steps 
# if config_parameters['apply_vc'] and not config_parameters['use_only_vc_data']:
#     num_train_epochs = num_train_epochs // 2

lr = config_parameters['learning_rate']


# Get current time using time module
current_time = time.strftime("%d%m%y_%H%M%S")
print("Current Time:", current_time)

# this is where the model will be saved
repo_name= "inprogress/" + '-'.join(
    [ 
        config_parameters['project'],
        model_name,
        str(config_parameters['learning_rate']),
        str(config_parameters['max_duration']),
        str(config_parameters['sample_size']),
        str(config_parameters['random_seed']),
        'vc' if config_parameters['apply_vc'] else 'n',
        'dr' if config_parameters['apply_dropout'] else 'n',
        current_time
    ]
)    

# create collator for padding
class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Prepare the batch dict in the format expected by the feature extractor
        batch = {
            input_features_key: [f[input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features]
        }
        
        # Use the feature extractor's native padding
        batch = self.feature_extractor.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )
        
        # Add labels
        batch["labels"] = torch.tensor(
            [f["label"] for f in features], 
            dtype=torch.long
        )
        
        return batch

data_collator = AudioDataCollator(feature_extractor)


# set training arguments
logger.info("Setting training arguments...")
training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,
    run_name='exp_' + repo_name, 
    report_to="wandb",  # enable logging to W&B
    logging_steps=1,  # how often to log to W&B
    per_device_train_batch_size=config_parameters['batch_size'], 
    per_device_eval_batch_size=8,    
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps", 
    save_steps=100,
    #save_strategy="epoch",
    learning_rate=config_parameters['learning_rate'],
    gradient_accumulation_steps=config_parameters['gradient_accumulation_steps'],
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    warmup_ratio=0.1, #0.1, 0.067, # 0.15 -- 0.1 works well
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,  # True if your metric should be maximized (like accuracy)
    save_total_limit=2,  # Keep only the best model
    dataloader_pin_memory=False,
    remove_unused_columns=True,
    gradient_checkpointing=True,
    fp16=True,
    push_to_hub=False,
)


# load evaluation metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(
        predictions=predictions, 
        references=eval_pred.label_ids
    )


trainer = Trainer(
    model,
    training_args,
    train_dataset=ADI5_sample_encoded["train"],
    eval_dataset=ADI5_sample_encoded["dev"],
    processing_class=feature_extractor,
    data_collator=data_collator,  
    compute_metrics=compute_metrics,
)

logger.info('Training model...')
trainer.train()

# save the model
logger.info('Saving model...')
trainer.save_model()

# evaluate the model
logger.info('Evaluating model...')
trainer.evaluate()


# clean up the cache files -- this should delete large arrow files from disk
logger.info('Cleaning up cache files...')
ADI5_sample_encoded.cleanup_cache_files()
