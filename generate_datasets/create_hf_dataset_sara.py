import os
os.environ['HF_HOME'] = 'hf_models'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/nethome/badr/projects/voice_conversion/wandb'

from datasets import Dataset, DatasetDict, Audio
import pandas as pd
import torchaudio
from pathlib import Path
from tqdm import tqdm

def create_audio_dataset(root_dir, target_sr=16000):
   data = []
   
   # Process 7sec files from each dialect folder
   for dialect in ['GLF', 'EGY', 'LEV']:
       dialect_path = os.path.join(root_dir, dialect, '7sec')
       if not os.path.exists(dialect_path):
           continue
           
       print(f'Processing {dialect} files...')
       for audio_file in tqdm(os.listdir(dialect_path)):
           if not audio_file.endswith('.wav'):
               continue
               
           file_path = os.path.join(dialect_path, audio_file)
           
           # Load and resample audio
           waveform, orig_sr = torchaudio.load(file_path)
           if orig_sr != target_sr:
               resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
               waveform = resampler(waveform)
               
           # Save resampled audio to new file
           new_path = os.path.join(root_dir, 'resampled', dialect, audio_file)
           os.makedirs(os.path.dirname(new_path), exist_ok=True)
           torchaudio.save(new_path, waveform, target_sr)
           
           # Get audio length in seconds
           length = waveform.shape[1] / target_sr
               
           data.append({
               'segment_id': os.path.splitext(audio_file)[0],
               'path': new_path,
               'audio': new_path, # datasets library will load this automatically
               'dialect': dialect,
               'length': length
           })
   
    # Create dataset from data
   df = pd.DataFrame(data)
   dataset = Dataset.from_pandas(df)
   # Cast the audio column to Audio feature with specific sampling rate
   dataset = dataset.cast_column('audio', Audio(sampling_rate=target_sr))
   
   # Create dataset dict with only test split
   dataset_dict = {'test': dataset}
   
   return DatasetDict(dataset_dict)


# Usage
dataset_name = 'sara-arabic-dialects'

# Create dataset
audio_path = "sara/SARA/dataset/"
dataset = create_audio_dataset(audio_path)

# Save to disk
dataset.save_to_disk(dataset_name)

# Verify the dataset
print(dataset['test'])
print(dataset['test'][0])