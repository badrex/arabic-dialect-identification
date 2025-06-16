#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['HF_HOME'] = '/nethome/badr/projects/voice_conversion/hf_models'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/nethome/badr/projects/voice_conversion/wandb'
os.environ['WANDB_DIR'] = '/nethome/badr/projects/voice_conversion/wandb'
os.environ['WANDB_CACHE_DIR'] = '/nethome/badr/projects/voice_conversion/wandb'
os.environ['WANDB_CONFIG_DIR'] = '/nethome/badr/projects/voice_conversion/wandb'
os.environ['WANDB_DATA_DIR'] = '/nethome/badr/projects/voice_conversion/wandb'
os.environ["WANDB_PROJECT"] = "text-adi5"

from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
from typing import Dict, List
import wandb


# login to wandb
wandb.login(key = "681a73d0905ff0e8f990610874a4b4837ae57255")

print("Starting...")
print(torch.__version__)
print(torch.version.cuda)

# set device to GPU if available
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")  # Set device to GPU
else:
    print("GPU is not available, using CPU instead")
    device = torch.device("cpu")  # Set device to CPU


def prepare_dataset(
    dataset_name: str,
    text_column: str,
    label_column: str,
    tokenizer,
    max_length: int = 128
):
    """
    Load and prepare dataset for training.
    """
    # Load dataset
    dataset = load_from_disk(dataset_name)
    
    # Get unique labels and create label2id mapping
    labels = sorted(set(dataset['train'][label_column]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    print(f"Labels: {labels}")
    print(f"Label mapping: {label2id}")
    
    def preprocess_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False  # We'll use dynamic padding with DataCollator
        )
        
        # Convert labels to ids
        tokenized["label"] = [label2id[label] for label in examples[label_column]]
        return tokenized
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset, label2id, id2label

def compute_metrics(pred):
    """
    Compute metrics for evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_dialect_classifier(
    dataset_name: str,
    text_column: str,
    label_column: str,
    output_dir: str,
    num_train_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 128
):
    """
    Fine-tune CamelBERT for dialect identification.
    """
    # Initialize tokenizer and model
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare dataset
    tokenized_dataset, label2id, id2label = prepare_dataset(
        dataset_name,
        text_column,
        label_column,
        tokenizer,
        max_length
    )
    
    # Initialize model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        #weight_decay=weight_decay,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps", 
        save_steps=300,
        report_to=["wandb"],  # Only report to wandb
        logging_steps=1,  # how often to log to W&B
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,  # Set to True if you want to push to Hub
        #load_best_model_at_end=True,
        #metric_for_best_model="accuracy",
        greater_is_better=True,  # True if your metric should be maximized (like accuracy)
        save_total_limit=1,  # Keep only the best model
        fp16=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate the model
    if "test" in tokenized_dataset:
        print("\nEvaluating on test set...")
        eval_results = trainer.evaluate(tokenized_dataset["test"])
        print(f"Test set results: {eval_results}")
    
    return trainer, model, tokenizer

# Example usage
if __name__ == "__main__":
    # Replace with your dataset details
    DATASET_NAME = "adi5-asr-transcriptions"
    TEXT_COLUMN = "transcription"
    LABEL_COLUMN = "dialect"
    OUTPUT_DIR = "models/finetune-arabic-BERT"
    
    trainer, model, tokenizer = train_dialect_classifier(
        dataset_name=DATASET_NAME,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )