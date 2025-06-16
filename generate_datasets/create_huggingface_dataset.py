#!/usr/bin/env python3
import os
from pathlib import Path
from datasets import Dataset, Audio, Features, Value
from tqdm.auto import tqdm
import argparse

class DialectDatasetCreator:
    def __init__(self, root_dir: str, output_dir: str, target_sr: int = 16000):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define domain mapping (lowercase to proper case)
        self.domain_mapping = {
            'radio': 'Radio',
            'tedx': 'TED Talks',
            'tv_drama': 'TV Dramas',
            'theatre': 'Theatre'
        }


    def create_dataset(self):
        """Create dataset with all features including audio"""
        all_files = []
        
        print("Collecting audio files and metadata...")
        
        # Walk through directory structure
        for domain_dir in self.root_dir.iterdir():
            if not domain_dir.is_dir():
                continue
                
            domain = self.domain_mapping.get(
                domain_dir.name.lower(), 
                domain_dir.name
            )
            
            for dialect_dir in domain_dir.iterdir():
                if not dialect_dir.is_dir():
                    continue
                    
                dialect = dialect_dir.name
                print(f"\nProcessing {domain} - {dialect}")
                
                # Process all MP3 files in dialect directory
                for audio_file in tqdm(list(dialect_dir.glob('*.mp3'))):
                    file_dict = {
                        'segment_id': audio_file.stem,
                        'path': str(audio_file),
                        'audio': str(audio_file),  # Audio feature will load this path
                        'dialect': dialect,
                        'domain': domain
                    }
                    all_files.append(file_dict)
        
        # Create initial dataset
        print("\nCreating initial dataset...")
        dataset = Dataset.from_list(all_files)
        
        # Cast to features without length first
        initial_features = Features({
            'segment_id': Value(dtype='string'),
            'path': Value(dtype='string'),
            'audio': Audio(sampling_rate=self.target_sr, mono=True),  # Changed: decode=False
            'dialect': Value(dtype='string'),
            'domain': Value(dtype='string')
        })
        
        print("Casting features and loading audio...")
        dataset = dataset.cast(initial_features)
        
        # Add length feature using the audio
        def add_length(example):
            # Get the audio info without decoding
            info = example['audio']
            # Calculate length from sampling rate and number of samples
            example['length'] = info['array'].shape[0] / info['sampling_rate']
            return example
        
        print("Calculating audio lengths...")
        dataset = dataset.map(add_length)
        
        return dataset

    def create_and_save_dataset(self, push_to_hub: bool = False, hub_path: str = None):
        """Create, save and optionally push dataset"""
        # Create dataset
        dataset = self.create_dataset()
        
        # Save dataset locally
        print(f"\nSaving dataset to {self.output_dir}...")
        dataset.save_to_disk(self.output_dir)
        print("Dataset saved successfully!")
        
        # Push to Hub if requested
        if push_to_hub and hub_path:
            print(f"\nPushing dataset to HuggingFace Hub at {hub_path}...")
            dataset.push_to_hub(hub_path)
            print("Dataset successfully pushed to Hub!")
        
        return dataset

def main():
    parser = argparse.ArgumentParser(description='Create HuggingFace dataset from Arabic dialect audio files')
    parser.add_argument('root_dir', help='Root directory containing audio files')
    parser.add_argument('output_dir', help='Output directory for the dataset')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Target sampling rate (default: 16000)')
    parser.add_argument('--push-to-hub', action='store_true',
                        help='Push dataset to HuggingFace Hub', 
                        default=False)
    parser.add_argument('--hub-path', type=str,
                        help='HuggingFace Hub path (e.g., "username/dataset-name")')
    
    args = parser.parse_args()
    
    # Create dataset
    creator = DialectDatasetCreator(args.root_dir, args.output_dir, args.sample_rate)
    dataset = creator.create_and_save_dataset(args.push_to_hub, args.hub_path)
    
    # Print dataset info
    print("\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    
    print("\nDialect distribution:")
    dialect_counts = dataset.unique('dialect')
    for dialect in dialect_counts:
        count = len(dataset.filter(lambda x: x['dialect'] == dialect))
        print(f"  {dialect}: {count}")
    
    print("\nDomain distribution:")
    domain_counts = dataset.unique('domain')
    for domain in domain_counts:
        count = len(dataset.filter(lambda x: x['domain'] == domain))
        print(f"  {domain}: {count}")
    
    # Calculate total audio duration
    total_duration = sum(dataset['length'])
    hours = total_duration / 3600
    print(f"\nTotal audio duration: {hours:.2f} hours")

    print("\nDataset saved to:", args.output_dir)
    
    # job done
    print("\nDone!")

if __name__ == "__main__":
    main()

