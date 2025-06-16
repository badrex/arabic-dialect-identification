import os
os.environ['HF_HOME'] = 'hf_models'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/data/users/babdullah/projects/voice_conversion/wandb'


from datasets import Dataset, DatasetDict, Audio
import pandas as pd
import torchaudio
from pathlib import Path
from tqdm import tqdm

dir_to_dialect = {
    'EGY': 'Egyptian',
    'GLF': 'Gulf',
    'LAV': 'Levantine',
    'MSA': 'MSA',
    'NOR': 'Maghrebi'
}


def create_audio_dataset(root_dir):

    # get all first level subdirectories

    splits = [
        d for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    print(f'Found {len(splits)} splits: {splits}')

    data = {
        split: [] for split in splits
    }
    
    for split in splits:
    #for split in ['train']:

        split_path = os.path.join(root_dir, split)

        for dialect in ['EGY', 'GLF', 'LAV', 'MSA', 'NOR']:

            dialect_path = os.path.join(split_path, dialect)
            if not os.path.exists(dialect_path):
                continue

            print(f'Processing {dialect} {split}...')     

            for audio_file in tqdm(os.listdir(dialect_path)):
                if not audio_file.endswith('.wav'):
                    continue

                #spkr_id = int(audio_file.split('.')[0].split('_')[-1])
                    
                file_path = os.path.join(dialect_path, audio_file)

                # Get audio length in seconds
                info = torchaudio.info(file_path)
                length = info.num_frames / info.sample_rate
                
                data[split].append({
                    'segment_id': os.path.splitext(audio_file)[0],
                    'path': file_path,
                    'audio': file_path,  # datasets library will load this automatically
                    'dialect': dir_to_dialect[dialect],
                    #'split': split,
                    #'speaker': spkr_id + 1,
                    'length': length
                })
                #break

    dataset_dict = DatasetDict({
        split: Dataset.from_pandas(
            pd.DataFrame(items)).cast_column('audio', Audio()) # Cast the audio column to Audio feature
            for split, items in data.items()
    })

    return dataset_dict
    
    # convert dict to DataFrame
    # df = pd.DataFrame(data)
    # dataset = Dataset.from_pandas(df)
    # # Cast the audio column to Audio feature
    # dataset = dataset.cast_column('audio', Audio())
    
    # return dataset

# Usage
#dataset = create_audio_dataset('converted_train_ADI5_12_spkrs_cv_ara/')
dataset = create_audio_dataset('ADI5-test')

dataset_name = 'adi5-aljazeera-arabic-dialects-test'

# Save to disk (optional)
dataset.save_to_disk(dataset_name)

print('Done!')