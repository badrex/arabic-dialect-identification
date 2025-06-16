#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['HF_HOME'] = '/nethome/badr/projects/voice_conversion/hf_models'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/nethome/badr/projects/voice_conversion/wandb'

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report
import joblib
from train_dialect_id_phonemes import ArabicDialectIdentifier


dir_to_dialect = {
    'EGY': 'Egyptian',
    'GLF': 'Gulf',
    'LEV': 'Levantine',
    'LAV': 'Levantine',
    'MSA': 'MSA',
    'NOR': 'Maghrebi'
}

dialect_to_dir = {v: k for k, v in dir_to_dialect.items()}

def evaluate_on_new_dataset(model_path, dataset_name, text_column, label_column):
    """
    Load a saved model and evaluate it on a new dataset.
    
    Args:
        model_path (str): Path to the saved model file
        dataset_name (str): Name of the Hugging Face dataset
        text_column (str): Name of the column containing text data
        label_column (str): Name of the column containing true labels
    
    Returns:
        tuple: (predictions, accuracy) if labels are provided, else just predictions
    """
    # Load the saved model
    print(f"Loading model from {model_path}")
    classifier = joblib.load(model_path)
    
    # Load the new dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_from_disk(dataset_name)
    
    # Get the test split
    if 'test' in dataset:
        eval_data = dataset['test']
    else:
        eval_data = dataset['train']  # Use whatever split is available

    # normalize dialect label LEV to LAV
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

    dataset_domains = eval_data.unique('domain')
    print(f"Dataset domains: {dataset_domains}")

    all_true_labels = []
    all_pred_labels = []


    for domain in dataset_domains:
        if domain == "Wikipedia":
            continue

        #print(f"Predicting dialects for domain: {domain}")

        domain_data = eval_data.filter(lambda example: example['domain'] == domain)
        domain_texts = domain_data[text_column]

        # making predictions for the domain
        domain_predictions = classifier.predict(domain_texts)
        domain_accuracy = accuracy_score(
            domain_data[label_column], 
            domain_predictions
        )

        all_true_labels.extend(domain_data[label_column])
        all_pred_labels.extend(domain_predictions)

        print(f"{domain} Acc.: {domain_accuracy:.4f}")

    # Get texts for prediction
    #texts = eval_data[text_column]
    
    # Make predictions
    #print("Making predictions...")
    #predictions = classifier.predict(texts)
    
    # If labels are provided, compute and print metrics
    #if label_column and label_column in eval_data.features:
    #true_labels = eval_data[label_column]
    
    total_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    
    print("\nEvaluation Results:")
    print(f"Total accuracy: {total_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_pred_labels))
        
    return all_pred_labels, total_accuracy


# Example usage
if __name__ == "__main__":
    # Replace these with your actual paths and dataset details
    MODEL_PATH = "models/phoneme_dialect_classifier.joblib"
    #DATASET_NAME = "madi5-0.2-phonemic-transcriptions"
    DATASET_NAME = "adi5-test-phonemic-transcriptions"
    TEXT_COLUMN = "phonemes"
    LABEL_COLUMN = "dialect"  # Set to None if no labels available
    
    # Evaluate the model
    predictions, accuracy = evaluate_on_new_dataset(
        model_path=MODEL_PATH,
        dataset_name=DATASET_NAME,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN
    )
    
    # Example of getting predictions with probabilities for specific examples
    # Load the model as an ArabicDialectIdentifier instance if you need probabilities
    classifier = joblib.load(MODEL_PATH)
    dataset = load_from_disk(DATASET_NAME)
    test_data = dataset['test'] if 'test' in dataset else dataset['train']

    # shuffle the dataset
    test_data = test_data.shuffle(seed=42)
    
    # Get probabilities for the first few examples
    texts = test_data[TEXT_COLUMN][:5]  # First 5 examples
    true_labels = test_data[LABEL_COLUMN][:5] if LABEL_COLUMN else None
    predictions_proba = classifier.predict_proba(texts)
    
    print("\nDetailed predictions for first 5 examples:")
    for text, probs, true_lable in zip(texts, predictions_proba, true_labels):
        pred_label = classifier.classes_[probs.argmax()]
        confidence = probs.max()
        print(f"\nText: {text}")
        print(f"Predicted Label: {pred_label}")
        print(f"True Label: {true_lable}")
        print(f"Confidence: {confidence:.4f}")