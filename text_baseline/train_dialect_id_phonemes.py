#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['HF_HOME'] = '/nethome/badr/projects/voice_conversion/hf_models'
os.environ[ 'NUMBA_CACHE_DIR'] = '/nethome/badr/projects/voice_conversion/wandb'

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import re
from datasets import load_from_disk
import joblib

from collections import defaultdict

PHONEME_TO_CHAR = {
    # Multi-character phonemes
    'ɑː': 'Ö',
    'aː': 'A',
    'eː': 'E',
    'iː': 'I',
    'oː': 'O',
    'uː': 'U',
    'əɜ': '€',
    'tʃ': 'Č',
    'dʒ': 'D',
    'ts': 'Z',
    'tɕ': 'J', 
    'ts.': 'L',
    't̪': 'T',
    'tʲ': 'N',
    'rʲ': 'R',
    's.': '$',
    'aɪ': '@',
    'aʊ': 'Q',
    'eɪ': 'Y',
    'oʊ': 'W',
    'oɪ': 'Ä',  
    'ɛ̃': 'V',
    'ɔ̃': '#',
    'ɑ̃': '8',
    'a5': '7',
    'i5': '6',
    'u5': '0',
    'ɑ5': '1',
    'ei5': '2',
    'ai5': '3',
    'iɛ5': '4',
    'onɡ5': '5',
    
    # Single-character phonemes (identical mapping)
    'a': 'a',
    'b': 'b',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'h': 'h',
    'i': 'i',
    'j': 'j',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    'q': 'q',
    'r': 'r',
    's': 's',
    't': 't',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'x': 'x',
    'y': 'y',
    'z': 'z',
    'ð': 'ð',
    'θ': 'θ',
    'æ': 'æ',
    'ç': 'ç',
    'ø': 'ø',
    'ŋ': 'ŋ',
    'œ': 'œ',
    'ɐ': 'ɐ',
    'ɑ': 'ɑ',
    'ɒ': 'ɒ',
    'ɔ': 'ɔ',
    'ɕ': 'ɕ',
    'ɛ': 'ɛ',
    'ɜ': 'ɜ',
    'ɟ': 'ɟ',
    'ɡ': 'ɡ',
    'ɣ': 'ɣ',
    'ɨ': 'ɨ',
    'ɪ': 'ɪ',
    'ɬ': 'ɬ',
    'ɲ': 'ɲ',
    'ɹ': 'ɹ',
    'ɾ': 'ɾ',
    'ʁ': 'ʁ',
    'ʃ': 'ʃ',
    'ʊ': 'ʊ',
    'ʌ': 'ʌ',
    'ʎ': 'ʎ',
    'ʏ': 'ʏ',
    'ʒ': 'ʒ',
    'ʔ': 'ʔ',
    'ʕ': 'ʕ',
    'ʉ': 'ʉ',
    'β': 'β',
    'ħ': 'ħ',
    'ə': 'ə',
    'ɚ': 'ɚ'
}

PHONEME_TO_CHAR = defaultdict(lambda: '', PHONEME_TO_CHAR)




class ArabicDialectIdentifier:
    def __init__(self, ngram_range=(1, 3), min_df=2, kernel='linear'):
        """
        Initialize the dialect identifier with customizable parameters.
        
        Args:
            ngram_range (tuple): Range of n-gram sizes (min_n, max_n)
            min_df (int): Minimum document frequency for features
            kernel (str): SVM kernel type ('linear', 'rbf', etc.)
        """
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                analyzer='char',
                ngram_range=ngram_range,
                lowercase=False,  # Keep phoneme cases as they are
                tokenizer=None,  # Use default character tokenization
                token_pattern=None,  # Not used with char analyzer
                min_df=min_df,
                preprocessor=self._preprocess_phonemes
            )),
            ('classifier', SVC(kernel=kernel, probability=True))
        ])
        
    def _preprocess_phonemes(self, text):
        """
        Preprocess phoneme sequences.
        
        Args:
            text (str): Input phoneme sequence
            
        Returns:
            str: Preprocessed phoneme sequence
        """
        # mapphonemes to single string representation
        text = ''.join([
            PHONEME_TO_CHAR[phoneme] for phoneme in text.split()
        ])

        #print(text[:10])

        return text
    
    
    def train(self, X, y):
        """
        Train the model on the provided data.
        
        Args:
            X (list): List of text samples
            y (list): List of corresponding dialect labels
        """
        self.pipeline.fit(X, y)
        
    def predict(self, X):
        """
        Predict dialects for new text samples.
        
        Args:
            X (list): List of text samples
            
        Returns:
            list: Predicted dialect labels
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each dialect.
        
        Args:
            X (list): List of text samples
            
        Returns:
            numpy.ndarray: Probability estimates for each class
        """
        return self.pipeline.predict_proba(X)
    
    def optimize_hyperparameters(self, X, y, cv=5):
        """
        Perform grid search to find optimal hyperparameters.
        
        Args:
            X (list): List of text samples
            y (list): List of corresponding dialect labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters found
        """
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'vectorizer__ngram_range': [(1, 2), (1, 3), (1, 4)],
            'vectorizer__min_df': [1, 2, 3],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='accuracy',
            verbose=5
        )
        
        grid_search.fit(X, y)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Update pipeline with best parameters
        self.pipeline = grid_search.best_estimator_
        return grid_search.best_params_
    
    # Add these methods to your ArabicDialectIdentifier class
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path where to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the pipeline
        joblib.dump(self.pipeline, path)
        print(f"Model saved to: {path}")

    def load_model(self, path):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        self.pipeline = joblib.load(path)
        print(f"Model loaded from: {path}")

def run_experiment(dataset_name, text_column, label_column):
    """
    Run the dialect identification experiment using a Hugging Face dataset.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face
        text_column (str): Name of the column containing text data
        label_column (str): Name of the column containing dialect labels
    """
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_from_disk(dataset_name)

    print(dataset)
    
    # Get train and validation splits
    train_data = dataset['train'].shuffle(seed=42)
    val_data = dataset['dev']
    
    # Extract features and labels
    X_train = train_data[text_column]#[:100]
    y_train = train_data[label_column]#[:100]
    X_val = val_data[text_column]
    y_val = val_data[label_column]

    # obtain a phoneme inventory from the phonemic transcriptions
    phoneme_inventory = set()

    for phoneme_transcription in X_train:
        phoneme_inventory.update(phoneme_transcription.split())

    for phoneme_transcription in X_val:
        phoneme_inventory.update(phoneme_transcription.split())

    print(f"Phoneme inventory: {phoneme_inventory}")
    print(f"Number of phonemes: {len(phoneme_inventory)}")

    # Check if any phonemes are missing from the mapping
    missing_phonemes = phoneme_inventory - set(PHONEME_TO_CHAR.keys())

    if missing_phonemes:
        print("Missing phonemes:", missing_phonemes)

    # Check if all characters are unique
    if len(set(PHONEME_TO_CHAR.values())) != len(PHONEME_TO_CHAR):
        print("Warning: Duplicate characters in mapping!")

    print("Total phonemes mapped:", len(PHONEME_TO_CHAR))
    

    print("\nDataset statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Initialize and train the model
    print("\nInitializing model...")
    classifier = ArabicDialectIdentifier()
    
    # Optional: Optimize hyperparameters
    print("\nOptimizing hyperparameters...")
    classifier.optimize_hyperparameters(X_train, y_train)
    
    # Train the model
    print("\nTraining model...")
    classifier.train(X_train, y_train)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = classifier.predict(X_val)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_val, y_pred))
    
    return classifier, accuracy

if __name__ == "__main__":
    # Example usage with a Hugging Face dataset
    DATASET_NAME = "adi5-asr-transcriptions"
    TEXT_COLUMN = "phonemes"
    LABEL_COLUMN = "dialect"
    
    classifier, accuracy = run_experiment(
        dataset_name=DATASET_NAME,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN
    )

    # Example usage:
    # Save the model after training
    classifier.save_model('models/phoneme_dialect_classifier.joblib')