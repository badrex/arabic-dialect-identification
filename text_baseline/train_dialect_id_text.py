#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['HF_HOME'] = '/nethome/badr/projects/voice_conversion/hf_models'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/nethome/badr/projects/voice_conversion/wandb'

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import re
from datasets import load_from_disk
import joblib


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
                min_df=min_df,
                preprocessor=self._preprocess_arabic
            )),
            ('classifier', SVC(kernel=kernel, probability=True))
        ])
        
    def _preprocess_arabic(self, text):
        """
        Preprocess Arabic text by removing diacritics and normalizing characters.
        
        Args:
            text (str): Input Arabic text
            
        Returns:
            str: Preprocessed text
        """
        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        
        # Normalize alef variations to bare alef
        text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
        
        # Normalize teh marbuta to heh
        text = text.replace('\u0629', '\u0647')
        
        # Remove non-Arabic characters
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        
        return text.strip()
    
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
    # Replace these parameters with your actual dataset details
    DATASET_NAME = "adi5-asr-transcriptions"
    TEXT_COLUMN = "transcription"
    LABEL_COLUMN = "dialect"
    
    classifier, accuracy = run_experiment(
        dataset_name=DATASET_NAME,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN
    )

    # Example usage:
    # Save the model after training
    classifier.save_model('models/dialect_classifier.joblib')