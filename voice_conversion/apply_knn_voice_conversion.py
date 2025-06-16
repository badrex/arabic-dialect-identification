# Description: 
# This script is used to take a collection of audio files (speech content)
# as input and apply voice conversion to a set of target voices
# using the KNN-VC model by Baas et al. (2023).

import os
import sys
import random
from collections import defaultdict
import argparse
import torch, torchaudio
from tqdm import trange, tqdm


# for debugging
print(torch.__version__)
print(torch.version.cuda)
print(sys.version)  # Full version info
print(sys.version_info[0:2])  # Major.minor version

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected!")

print(torch.cuda.is_available())
print(torch.cuda.device_count())


# specify split --- usually only the train split is converted
split = "train"

# get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# save all converted segments into this dir as wav files
output_dir = os.path.join(script_dir, f"converted_{split}_ADI5_6_spkr", split)

# create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create subdirectories for each dialect
dialects = ['GLF', 'EGY', 'LAV', 'NOR', 'MSA']
for dialect in dialects:
    dialect_dir = os.path.join(output_dir, dialect)  

    if not os.path.exists(dialect_dir):
        os.makedirs(dialect_dir)

        print(f"Created directory: {dialect_dir}")


# load the voice conversion model from disk 
knn_vc = torch.hub.load(
    'bshall/knn-vc', 
    'knn_vc', 
    prematched=True, 
    trust_repo=True, 
    pretrained=True, 
    device='cuda'
)


# load the target audio files
target_voices_dir = 'targets_6/' 

# read all files in the above dir
target_voice_files = os.listdir(target_voices_dir)


voice_to_data = defaultdict(lambda: defaultdict())
male_voices, female_voices, all_target_voices = [], [], []

for target_file in target_voice_files:
    # get gender info from file name --- useful for some experiments 
    gender = target_file[6].upper()
    voice_name = target_file[:-4].upper()

    if gender == 'M':
        male_voices.append(voice_name)

    elif gender == 'F':
        female_voices.append(voice_name)

    all_target_voices.append(voice_name)

    voice_to_data[voice_name]['gender'] = gender
    voice_to_data[voice_name]['path'] = target_voices_dir + target_file


# extract the features of the target audio files
target_voice_features = defaultdict()

for target_voice in voice_to_data:
    target_voice_features[target_voice] = knn_vc.get_matching_set(
        [voice_to_data[target_voice]['path'], ]
    )


# main loop here for voice conversion
# for each dialect, convert all the audio files in the dialect folder
for dialect in ['GLF', 'EGY', 'LAV', 'NOR', 'MSA']:
    # load the source audio files 
    dialect_dir = f"ADI5_samples/{split}/" + dialect + '/'

    audio_clips_to_convert = os.listdir(dialect_dir)

    audio_clip_path = defaultdict()

    for i, audio_clip in enumerate(sorted(audio_clips_to_convert), start=1):
        audio_clip_id = audio_clip[:-4].upper()
        audio_clip_path[audio_clip_id] = dialect_dir + audio_clip


    # extract the features of the training audio files
    # audio_clip_features = defaultdict()

    # for audio_clip in tqdm(audio_clip_dir):
    #     audio_clip_features[audio_clip] = knn_vc.get_features(
    #         audio_clip_dir[audio_clip]['path']
    #     )


    # convert each audio clip in the dialect folder using the KNN-VC model
    # then save to output_dir as a wav file
    for audio_clip in tqdm(audio_clip_path):

        print(dialect, audio_clip)

        # extract the features of the audio sample we want to convert
        audio_clip_features = knn_vc.get_features(
            audio_clip_path[audio_clip]
        )

        # loop over all target voices 
        # NOTE: only in case we want to convert to multiple target voices
        # for i, target_voice in enumerate(all_target_voices):

        #     print("I am converting", audio_clip, "to", target_voice)

        #     converted_audio_id = audio_clip + "_" + str(i)


        #     #print("Audio shape:", audio_clip_features.shape)
        #     #print("Audio shape:", target_voice_features[target_voice].shape)

        #     converted_audio_sample = knn_vc.match(
        #         audio_clip_features, 
        #         target_voice_features[target_voice], 
        #         topk=4
        #     )

        #     #print("Audio shape:", converted_audio_sample.shape)
        #     #print("Audio dtype:", converted_audio_sample.dtype)

        #     # save the converted clip as wav files
        #     output_path = output_dir + dialect + '/' + converted_audio_id + '.wav'

        #     torchaudio.save(              
        #         output_path,
        #         converted_audio_sample[None],
        #         16000
        #     )

        # if we want to convert to a single target voice, use sampling 
        target_voice = random.choice(all_target_voices)

        print("I am converting", audio_clip, "to", target_voice)

        target_voice_id = all_target_voices.index(target_voice)

        converted_audio_id = audio_clip + "_" + str(target_voice_id)

        converted_audio_sample = knn_vc.match(
            audio_clip_features, 
            target_voice_features[target_voice], 
            topk=4
        )

        #print("Audio shape:", converted_audio_sample.shape)
        #print("Audio dtype:", converted_audio_sample.dtype)

        # save the converted clip as wav files
        output_path = os.path.join(output_dir, dialect) + '/' + converted_audio_id + '.wav'

        torchaudio.save(              
            output_path,
            converted_audio_sample[None],
            16000
        )

print("Done")

