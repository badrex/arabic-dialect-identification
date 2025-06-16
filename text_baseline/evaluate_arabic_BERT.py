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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import classification_report, accuracy_score
import torch
import numpy as np


dir_to_dialect = {
    'EGY': 'Egyptian',
    'GLF': 'Gulf',
    'LEV': 'Levantine',
    'LAV': 'Levantine',
    'MSA': 'MSA',
    'NOR': 'Maghrebi'
}

dialect_to_dir = {v: k for k, v in dir_to_dialect.items()}


def load_model_and_tokenizer(model_path):
    """
    Load the fine-tuned model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def predict_dialects(texts, model, tokenizer, max_length=128, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict dialects for a list of texts.
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Lists to store predictions and probabilities
    all_predictions = []
    all_probabilities = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert numeric predictions to labels
    id2label = model.config.id2label
    predicted_labels = [id2label[pred] for pred in all_predictions]
    
    return predicted_labels, np.array(all_probabilities)

def evaluate_on_dataset(dataset_name, text_column, label_column, model_path, batch_size=32):
    """
    Evaluate the model on a dataset.
    """
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_from_disk(dataset_name)
    
    # Get the appropriate split
    eval_data = dataset['test']

    # Normalize dialect label LEV to LAV
    eval_data = eval_data.map(
        lambda example: {
            **example,
            label_column: "LAV" if example[label_column] == "LEV" else example[label_column]
        }
    )

    dialects = eval_data.unique(label_column)

    # print unique dialects 
    print(f"Unique dialects: {dialects}")

    if "Gulf" in dialects:
        # map dialects to directories
        eval_data = eval_data.map(
            lambda example: {
                **example,
                label_column: dialect_to_dir[example[label_column]]
            }
        )


    # check if domain is not one of the features of the dataset
    if eval_data.features.get('domain') is None:
        # add domain to the dataset
        eval_data = eval_data.map(
            lambda example: {
                **example,
                'domain': 'In-domain'
            }
        )

    # get accuracy for each domain
    dataset_domains = eval_data.unique('domain')

    print(f"Dataset domains: {dataset_domains}")

    all_true_labels = []
    all_pred_labels = []
    all_probabilities = []

    for domain in dataset_domains:
        # skip wikipedia domain
        if domain == "Wikipedia":
            continue

        domain_data = eval_data.filter(lambda example: example['domain'] == domain)

        domain_texts = domain_data[text_column]
        domain_labels = domain_data[label_column]

        # making predictions for the domain
        domain_predictions, probs = predict_dialects(domain_texts, model, tokenizer, batch_size=batch_size)
        domain_accuracy = accuracy_score(domain_labels, domain_predictions)

        all_true_labels.extend(domain_labels)
        all_pred_labels.extend(domain_predictions)
        all_probabilities.extend(probs)

        print(f"{domain} Acc.: {domain_accuracy:.4f}")

    # Get overall metrics
    print("\nOverall Metrics:")
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_pred_labels))

    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}")

    return all_pred_labels, all_probabilities, accuracy


if __name__ == "__main__":
    # Replace these with your actual paths and dataset details
    MODEL_PATH = "models/finetune-arabic-BERT"
    #DATASET_NAME = "madi5-0.2-asr-transcriptions"
    DATASET_NAME = "adi5-test-asr-transcriptions"
    TEXT_COLUMN = "transcription"
    LABEL_COLUMN = "dialect"  # Set to None if no labels available
    
    # Evaluate on dataset
    predictions, probabilities, accuracy = evaluate_on_dataset(
        dataset_name=DATASET_NAME,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        model_path=MODEL_PATH,
        batch_size=32
    )
    
    # Example of getting predictions for specific examples
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Example texts
    example_texts = [
        "كيف حالك اليوم",
        "عامل ايه النهاردة",
        "شلونك اليوم"
    ]
    
    predictions, probs = predict_dialects(example_texts, model, tokenizer)
    
    print("\nExample Predictions:")
    for text, pred, prob in zip(example_texts, predictions, probs):
        confidence = np.max(prob)
        print(f"\nText: {text}")
        print(f"Predicted Dialect: {pred}")
        print(f"Confidence: {confidence:.4f}")