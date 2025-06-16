import os
import sys
import random
import glob
from pathlib import Path
from collections import defaultdict
import argparse
import math
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from tqdm import trange, tqdm

import librosa
#import matplotlib.pyplot as plt
#from IPython.display import Audio
#from torchaudio.utils import download_asset

# for debugging
print(torch.__version__)
print(torch.version.cuda)
print(sys.version)  # Full version info
print(sys.version_info[0:2])  # Major.minor version

project_dir = "/nethome/badr/projects/voice_conversion/"

# specify the type of augmentation to apply
# TODO: make this a command line argument
# Possible augmentations:
# "RIR", "PITCH_SHIFT", "ADDITIVE_NOISE", "SPEC_AUG",
augmentation = "PITCH_SHIFT"

SPEC_AUGMENTATIONS = {
    "TIME_STRETCH",
    "TIME_MASK",
    "FREQUENCY_MASK"
}

TIME_STRETCH_RATES = {0.7, 0.8, 1.2, 1.3, 1.4, 1.5}
PITSH_SHIFT_STEPS = {-8, -6, -4, -2, 2, 4, 6, 8}

if augmentation == "RIR":

    # specify directories for RIRs

    print("Read RIR files ...")

    # specify directories for audio that simulate room impulse responses (RIRs)
    rir_dirs = [
        "sounds_and_noise/RIRS_NOISES/pointsource_noises", 
        "sounds_and_noise/RIRS_NOISES/real_rirs_isotropic_noises",
        #"sounds_and_noise/RIRS_NOISES/simulated_rirs/largeroom",
        # "sounds_and_noise/RIRS_NOISES/simulated_rirs/mediumroom",
        # "sounds_and_noise/RIRS_NOISES/simulated_rirs/smallroom",
    ]

    # get paths for RIR wav files 
    RIR_FILES = []

    RIR_FILES.extend(
        str(p) 
        for rir_dir in rir_dirs
        for p in list(Path(rir_dir).rglob("*.wav"))
    )

    print(f"Num of RIR files :{len(RIR_FILES)}")


# print("CUDA Available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Number of GPUs:", torch.cuda.device_count())
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("No GPUs detected!")

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())


# functions for data augmentation
def process_audio_file(file_path, resample=None):
    """
    Process audio file with sox effects
    """
    effects = [["remix", "1"]]  # Convert to mono
    if resample:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f"{resample}"]
        ])
    return torchaudio.sox_effects.apply_effects_file(file_path, effects=effects)


def create_spectrogram(waveform, power=1.0):
    """
    Create spectrogram from waveform
    """
    spectrogram = T.Spectrogram(
        n_fft=400,
        win_length=None,
        hop_length=None,
        center=True,
        pad_mode="reflect",
        power=power
    )
    return spectrogram(waveform)


def reconstruct_waveform(spec, rate=16000):
    """
    Convert spectrogram back to audio waveform
    """
    # Convert complex spectrogram to magnitude spectrogram for GriffinLim
    if torch.is_complex(spec):
        spec = torch.abs(spec)
    
    griffin_lim = T.GriffinLim(
        n_fft=400,
        n_iter=32,
        win_length=None,
        hop_length=None,
        power=1.0  # Important: use power=1.0 with magnitude spectrogram
    )
    
    # Reconstruct audio
    waveform = griffin_lim(spec)
    return waveform

def stretch_audio(audio_path, stretch_rate=1.2):
    """
    Load audio, create spectrogram, and apply time stretching
    """
    # 1. Load and process audio
    waveform, _ = process_audio_file(audio_path)
    
    # 2. Create complex spectrogram (power=None for time stretching)
    spec = create_spectrogram(waveform, power=None)
    
    # 3. Apply time stretching
    stretch = T.TimeStretch()
    stretched_spec = stretch(spec, overriding_rate=stretch_rate)
    
    # 4. Convert to waveform and return
    return reconstruct_waveform(stretched_spec)


def mask_frequency(audio_path, freq_mask_param=160):
    """
    Apply frequency masking to spectrogram
    """
    # set random seed 
    torch.random.manual_seed(42)

    # 1. Load and process audio
    waveform, _ = process_audio_file(audio_path)

    # 2. Create complex spectrogram
    spec = create_spectrogram(waveform)

    # 3. Apply frequency masking
    freq_masking = T.FrequencyMasking(freq_mask_param)

    # 4. Apply mask
    freq_masked_spec = freq_masking(spec)

    # 5. Convert to waveform and return
    return reconstruct_waveform(freq_masked_spec)


def mask_time(audio_path, time_mask_param=80):
    """
    Apply time masking to spectrogram
    """
    # set random seed 
    torch.random.manual_seed(42)

    # 1. Load and process audio
    waveform, _ = process_audio_file(audio_path)

    # 2. Create complex spectrogram
    spec = create_spectrogram(waveform)

    # 3. Apply time masking
    time_masking = T.TimeMasking(time_mask_param)

    # 4. Apply mask
    time_masked_spec = time_masking(spec)

    # 5. Convert to waveform and return
    return reconstruct_waveform(time_masked_spec)


def pitch_shift(audio_path, pitch_shift_param):
    """
    Apply pitch shifting to audio
    """
    # 1. Load and process audio
    waveform, _ = process_audio_file(audio_path)

    # 2. Apply pitch shifting
    pitch_shift = T.PitchShift(sample_rate=16000, n_steps=pitch_shift_param)

    # 3. Apply pitch shift
    shifted_waveform = pitch_shift(waveform)

    # 4. Return shifted waveform
    return shifted_waveform


def apply_rir(audio_path):
    """
    Given an audio and RIR, apply room impulse response to audio
    """
    # 1. Load and process audio
    waveform, _ = process_audio_file(audio_path)

    #print(audio_path, rir_path)

    # this was a quick and dirty solution around the issue of
    # some RIR files not being loaded, it is not the best solution
    # TODO: find a better solution, and even better, prevent the issue
    it_works = False

    while not it_works:
        # 2. Sample a RIR wav
        rir_path=random.choice(RIR_FILES)

        try:
            # 3. Load and process RIR with specific parameters
            rir_raw, sample_rate = torchaudio.load(
                rir_path,
                normalize=True,  # Normalize the audio
                channels_first=True,  # Ensure channels are first
            )
            it_works = True

        except RuntimeError as e:
            #print("Error loading RIR file: ", rir_path)
            #print(e)
            #print("Trying another RIR file ...")
            # try another RIR file
            it_works = False
            
    # Process RIR
    rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    
    # Apply convolution
    augmented = F.fftconvolve(waveform, rir)
    return augmented

# specify directories for additive noise (MUSAN dataset)
NOISE_TYPE_DIRS = [
    'sounds_and_noise/musan/noise/free-sound',
    'sounds_and_noise/musan/noise/sound-bible',
    'sounds_and_noise/musan/music/jamendo',
    'sounds_and_noise/musan/music/hd-classical',
    'sounds_and_noise/musan/music/rfm',
    'sounds_and_noise/musan/music/fma-western-art',
    'sounds_and_noise/musan/music/fma'
]


class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'], # convert to mono
            ['rate', str(self.sample_rate)], # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise ) / 2

# create a list of noise transforms
if augmentation == "ADDITIVE_NOISE":

    print("Creating noise transforms functions ...")
    NOISE_TRANSFORMS = [
        RandomBackgroundNoise(16000, add_noise_dir)
        for add_noise_dir in NOISE_TYPE_DIRS
    ]

    for add_noise_dir in NOISE_TYPE_DIRS:

        # define a RandomBackgroundNoise object for each noise type
        add_noise = RandomBackgroundNoise(16000, add_noise_dir)

        # add function to a data structure
        NOISE_TRANSFORMS.append(add_noise)

# specify split --- usually only the train split is converted
split = "train"

# save all converted segments into this dir as wav files
output_dir = f"ADI5_augmented_{split}_{augmentation.lower()}/"

# create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    # create subdirectories for each dialect
    for dialect in ['GLF', 'EGY', 'LAV', 'NOR', 'MSA']:
        dialect_output_dir = os.path.join(output_dir, dialect)
        if not os.path.exists(dialect_output_dir):
            os.makedirs(dialect_output_dir)

# main loop here for voice conversion
# for each dialect, convert all the audio files in the dialect folder
for dialect in ['GLF', 'EGY', 'LAV', 'NOR', 'MSA']:
    # load the source audio files 
    # here this assunes that the audio files are stored in a directory like 
    # ADI5/{split}/{dialect}/file_id.wav
    dialect_dir = f"ADI5_samples/{split}/{dialect}/"

    audio_clips_to_process = os.listdir(dialect_dir)

    audio_clip_path = defaultdict()

    for i, audio_clip in tqdm(
        enumerate(sorted(audio_clips_to_process)), 
        total=len(audio_clips_to_process),
        dynamic_ncols=True
    ):
        audio_clip_id = audio_clip[:-4].upper()
        audio_clip_path[audio_clip_id] = dialect_dir + audio_clip

        # if SPEC_AUG, sample an augmentation technique
        if augmentation == "SPEC_AUG":
            # randomly sample an augmentation technique
            # from the set of available augmentations
            spec_aug = random.choice(list(SPEC_AUGMENTATIONS))

            if spec_aug == "TIME_STRETCH":
                # create streched audio
                processed_audio = stretch_audio(
                    audio_clip_path[audio_clip_id], 
                    stretch_rate=random.choice(list(TIME_STRETCH_RATES))
                )
            elif spec_aug == "TIME_MASK":
                processed_audio = mask_time(audio_clip_path[audio_clip_id])

            elif spec_aug == "FREQUENCY_MASK":
                processed_audio = mask_frequency(audio_clip_path[audio_clip_id])

        elif augmentation == "PITCH_SHIFT":
            # randomly sample pitch shift steps
            n_steps = random.choice(list(PITSH_SHIFT_STEPS))

            # create pitch shifted audio
            processed_audio = pitch_shift(
                audio_clip_path[audio_clip_id], 
                n_steps
            )

        elif augmentation == "RIR":
            # apply room impulse response
            processed_audio = apply_rir(
                audio_clip_path[audio_clip_id],
            )

        elif augmentation == "ADDITIVE_NOISE":
            # appy additive noise to the audio
            # randomly sample a noise type

            noise_transform = random.choice(NOISE_TRANSFORMS)
            waveform, _ = process_audio_file(audio_clip_path[audio_clip_id])
            processed_audio = noise_transform(waveform)

        # save the spectrogram as a wav file
        output_path = output_dir + dialect + '/' + audio_clip_id + '.wav'

        #print(processed_audio.shape)

        torchaudio.save(output_path, processed_audio, 16000)



