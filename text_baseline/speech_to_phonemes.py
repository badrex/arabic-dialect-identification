#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['HF_HOME'] = '/nethome/badr/projects/voice_conversion/hf_models'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/nethome/badr/projects/voice_conversion/wandb'

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_from_disk, Dataset, DatasetDict
from collections import defaultdict 
from tqdm import tqdm
import pandas as pd

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


# load model and processor
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
)

phonemizer_model = Wav2Vec2ForCTC.from_pretrained(
   "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
).to(device)


# load dummy dataset and read soundfiles
# ds = load_from_disk("adi5-aljazeera-arabic-dialects")['dev']

# print(ds[0])

# # tokenize
# input_values = processor(
#     ds[0]["audio"]["array"], 
#     return_tensors="pt",
#     sampling_rate=16000
# ).input_values

# # retrieve logits
# with torch.no_grad():
#     logits = model(input_values).logits

# # take argmax and decode
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)

# print(transcription)
# => should give ['m ɪ s t ɚ k w ɪ l t ɚ ɪ z ð ɪ ɐ p ɑː s əl l ʌ v ð ə m ɪ d əl k l æ s ɪ z æ n d w iː aʊ ɡ l æ d t ə w ɛ l k ə m h ɪ z ɡ ɑː s p ə']

# in-domaon data
#dataset_name = "adi5-aljazeera-arabic-dialects"

# cross-domain data
#dataset_name = "arabic-dialect-identification/MADI-5-Eval"
#dataset_name = "adi5-aljazeera-arabic-dialects-test"
dataset_name = "MADI-5-Eval-0.2"

print(f"Loading the dataset {dataset_name} ...")

if dataset_name == "adi5-aljazeera-arabic-dialects":
    input_audio_dataset = load_from_disk(dataset_name)
    splits = ["dev", "train"]
    output_dataset = "adi5-phonemic-transcriptions"

elif dataset_name == "adi5-aljazeera-arabic-dialects-test":
    input_audio_dataset = load_from_disk(dataset_name)
    splits = ["test"]
    output_dataset = "adi5-test-phonemic-transcriptions"

elif dataset_name == "MADI-5-Eval-0.2":
    input_audio_dataset = load_from_disk(dataset_name)
    input_audio_dataset = DatasetDict({
        'test': input_audio_dataset  # Move all data to test split
    })
    
    splits = ["test"]
    output_dataset = "madi5-0.2-phonemic-transcriptions"


# filter out data points that are shorter than 1 second in length 
input_audio_dataset = input_audio_dataset.filter(lambda x: x['length'] > 1)

new_dataset = {
    split: [] for split in splits
}

for split in splits:

    # load the dataset split
    audio_dataset = input_audio_dataset[split]

    # create a dict for the dataset
    segment_to_data = defaultdict(lambda: defaultdict())

    audio_arrays = []
    segment_ids = []

    # this is too slow -- refactor
    # for i, sample in tqdm(enumerate(audio_dataset), total=len(audio_dataset)):

    #     print(f"Processing sample {i}")
    #     segment_ids.append(sample["segment_id"])
    #     audio_arrays.append(sample["audio"]["array"])

    #     segment_id = sample["segment_id"]
    #     dialect = sample["dialect"]
    #     length = sample["length"]
    
    #     segment_to_data[segment_id]["dialect"] = dialect
    #     #segment_to_data[segment_id]["audio"] = audio
    #     segment_to_data[segment_id]["length"] = length

    # flatten the dataset
    print("Flattening the dataset...")
    segment_ids = audio_dataset['segment_id']
    dialects = audio_dataset['dialect']
    lengths = audio_dataset['length']
    
    # audio_arrays = [
    #     audio_dataset[i]['audio']["array"] for i in range(len(audio_dataset))
    # ]

    segment_to_data = {
        segment_id: {
            'dialect': dialect,
            'length': length,
            'phonemes': None # to be replaced by ASR output
        }
        for segment_id, dialect, length in zip(segment_ids, dialects, lengths)
    }

    if dataset_name == "MADI-5-Eval-0.2":
        domains = audio_dataset['domain'] 

        # add the entry domain to each item segment_to_data dict
        for segment_id, domain in zip(segment_ids, domains):
            segment_to_data[segment_id]['domain'] = domain


        # segment_to_data = { 
        #     segment_id: {
        #         'domain': domain
        #     }
        #     for segment_id, domain in zip(segment_ids, domains)
        # }

    for i, sample in tqdm(enumerate(audio_dataset), total=len(audio_dataset)):
        segment_id = sample["segment_id"]
        print(f"Processing sample {i}: {segment_id}")
        audio_arrays.append(sample["audio"]["array"])

    # obtain a list of all the audio files and labels in the dataset
    #segments_ids = list(segment_to_data.keys())

    # audio_arrays = [
    #     segment_to_data[segment_id]['audio'] for segment_id in segments_ids
    # ] 

    print(f"Loaded {len(segment_ids)} segments")
    print(f"Number of audio arrays: {len(audio_arrays)}")


    # loop over samples in the dataset 
    for i in range(len(segment_ids)):

        segment_id = segment_ids[i]

        # tokenize
        input_values = processor(
            audio_arrays[i], 
            return_tensors="pt",
            sampling_rate=16000
        ).input_values.to(device)

        # retrieve logits
        with torch.inference_mode():
            logits = phonemizer_model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(predicted_ids)

        print(f"{i:<5}, "
              f"{segment_to_data[segment_id]['dialect']}, " 
              f"{segment_id}")
        print(f"{' '.join(phonemes)}")

        new_dataset[split].append({
            'segment_id': segment_id,
            'dialect': segment_to_data[segment_id]['dialect'],
            'length': segment_to_data[segment_id]['length'],
            'phonemes': ' '.join(phonemes)
        })

        # add domain to the data point if multi-domain
        if dataset_name == "MADI-5-Eval-0.2":
            new_dataset[split][-1]['domain'] = segment_to_data[segment_id]['domain']

        # for debgging purposes
        #break

# save the the segment_to_data dict to disk as a huggingface dataset
asr_dataset = DatasetDict({
    split: Dataset.from_pandas(
        pd.DataFrame(items)
    )
    for split, items in new_dataset.items()
})

print(asr_dataset)

# print first sample of train split
print(asr_dataset[split][0])

asr_dataset.save_to_disk(output_dataset)

print("Done!")