from datasets import load_dataset
import pandas as pd
import numpy as np
import torchaudio
import torch
from pathlib import Path
import os

def download_and_prepare_data():
    """Download and prepare Common Voice Arabic data"""
    # Load dataset
    dataset = load_dataset("mozilla-foundation/common_voice_17", "ar", split=['train', 'validation'])
    
    # Create directories for audio files
    Path('audio_files/train').mkdir(parents=True, exist_ok=True)
    Path('audio_files/dev').mkdir(parents=True, exist_ok=True)
    
    # Process train and validation sets
    train_clips = []
    train_speakers = {}
    dev_clips = []
    dev_speakers = {}
    
    # Process training set
    for idx, item in enumerate(dataset[0]):
        speaker_id = item['client_id']
        audio_path = f"audio_files/train/{idx}.wav"
        
        # Save audio file
        item['audio']['array'].save(audio_path)
        duration = len(item['audio']['array']) / item['audio']['sampling_rate']
        
        # Update clips data
        clip_info = {
            'speaker_id': speaker_id,
            'original_speaker_id': speaker_id,
            'clip_id': idx,
            'original_path': item['path'],
            'output_path': audio_path,
            'duration': duration,
            'text': item['sentence']
        }
        train_clips.append(clip_info)
        
        # Update speaker data
        if speaker_id not in train_speakers:
            train_speakers[speaker_id] = {'total_duration': 0, 'num_clips': 0}
        train_speakers[speaker_id]['total_duration'] += duration
        train_speakers[speaker_id]['num_clips'] += 1
    
    # Process validation set (similar to training)
    for idx, item in enumerate(dataset[1]):
        speaker_id = item['client_id']
        audio_path = f"audio_files/dev/{idx}.wav"
        
        # Save audio file
        item['audio']['array'].save(audio_path)
        duration = len(item['audio']['array']) / item['audio']['sampling_rate']
        
        # Update clips data
        clip_info = {
            'speaker_id': speaker_id,
            'original_speaker_id': speaker_id,
            'clip_id': idx,
            'original_path': item['path'],
            'output_path': audio_path,
            'duration': duration,
            'text': item['sentence']
        }
        dev_clips.append(clip_info)
        
        # Update speaker data
        if speaker_id not in dev_speakers:
            dev_speakers[speaker_id] = {'total_duration': 0, 'num_clips': 0}
        dev_speakers[speaker_id]['total_duration'] += duration
        dev_speakers[speaker_id]['num_clips'] += 1
    
    # Convert to DataFrames and save
    train_clips_df = pd.DataFrame(train_clips)
    dev_clips_df = pd.DataFrame(dev_clips)
    
    train_speakers_df = pd.DataFrame([
        {'speaker_id': k, 'original_speaker_id': k, 'total_duration': v['total_duration'], 'num_clips': v['num_clips']}
        for k, v in train_speakers.items()
    ])
    
    dev_speakers_df = pd.DataFrame([
        {'speaker_id': k, 'original_speaker_id': k, 'total_duration': v['total_duration'], 'num_clips': v['num_clips']}
        for k, v in dev_speakers.items()
    ])
    
    # Save metadata
    train_clips_df.to_csv('clips_metadata_train.csv', index=False)
    dev_clips_df.to_csv('clips_metadata_dev.csv', index=False)
    train_speakers_df.to_csv('speakers_metadata_train.csv', index=False)
    dev_speakers_df.to_csv('speakers_metadata_dev.csv', index=False)
    
    return train_clips_df, dev_clips_df, train_speakers_df, dev_speakers_df

# The rest of the concatenation code remains the same as in the previous script...
# [Previous concatenation functions here]

def main():
    # First download and prepare the data
    print("Downloading and preparing Common Voice Arabic data...")
    train_clips, dev_clips, train_speakers, dev_speakers = download_and_prepare_data()
    
    # Then proceed with concatenation
    print("Creating concatenated audio files...")
    results = concatenate_audio_files(num_speakers=60, min_duration=65)
    
    return results

if __name__ == "__main__":
    results = main()