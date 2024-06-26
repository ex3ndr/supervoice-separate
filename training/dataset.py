import torch
import torchaudio
from supervoice_separate.config import config
from supervoice_separate.audio import load_mono_audio, spectogram
from .effects import light_noisy_pipeline, light_noisy_voiced_pipeline, light_noisy_common_pipeline
from .distorter import create_distorter
from pathlib import Path
import random

#
# Samplers
#

def load_clean_sampler(datasets, duration, return_source = False):

    # Target duration
    frames = int(duration * config.audio.sample_rate)

    # Load the datasets
    files = []
    if isinstance(datasets, str):
        with open(datasets + "files_all.txt", 'r') as file:
            dataset_files = file.read().splitlines()
        dataset_files = [datasets + p + ".flac" for p in dataset_files]
    else:
        dataset_files = []
        for dataset in datasets:
            dataset_files += list(Path(dataset).rglob("*.wav")) + list(Path(dataset).rglob("*.flac"))
        dataset_files = [str(p) for p in dataset_files]
    files += dataset_files
    print(f"Loaded {len(files)} files")

    # Sample a single item
    def sample_item():

        # Load random audio
        f = random.choice(files)
        audio = load_mono_audio(f, config.audio.sample_rate)

        # Pad or trim audio
        if audio.shape[0] < frames:
            padding = frames - audio.shape[0]
            padding_left = random.randint(0, padding)
            padding_right = padding - padding_left
            audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)
        else:
            start = random.randint(0, audio.shape[0] - frames)
            audio = audio[start:start + frames]

        # Spectogram
        spec = spectogram(audio, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        # Return result
        if return_source:
            return spec, audio
        else:
            return spec

    return sample_item

def load_effected_sampler(datasets, effect, duration, return_source = False):

    # Target duration
    frames = int(duration * config.audio.sample_rate)

    # Load the datasets
    files = []
    for dataset in datasets:
        dataset_files = list(Path(dataset).rglob("*.wav")) + list(Path(dataset).rglob("*.flac"))
        dataset_files = [str(p) for p in dataset_files]
        files += dataset_files
    
    # Sample a single item
    def sample_item():

        # Load random audio
        f = random.choice(files)
        audio = load_mono_audio(f, config.audio.sample_rate)

        # Pad or trim audio
        if audio.shape[0] < frames:
            padding = frames - audio.shape[0]
            padding_left = random.randint(0, padding)
            padding_right = padding - padding_left
            audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)
        else:
            start = random.randint(0, audio.shape[0] - frames)
            audio = audio[start:start + frames]

        # Apply effect
        audio_with_effect = effect(audio)

        # Spectogram
        spec = spectogram(audio, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        # Spectogram with effect
        spec_with_effect = spectogram(audio_with_effect, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        # Return results
        if return_source:
            return (spec, spec_with_effect, audio, audio_with_effect)
        else:
            return (spec, spec_with_effect)

    # Return generator
    return sample_item


def load_distorted_sampler(datasets, duration, return_source = False):

    # Load RIR files
    rir_files = []
    with open('./external_datasets/rir-1/files.txt', 'r') as file:
        for line in file:
            rir_files.append("./external_datasets/rir-1/" + line.strip())

    # Load BG files
    bg_files = []
    for p in Path("./external_datasets/dns-noise").rglob("*.wav"):
        bg_files.append(str(p))
    
    # Distorter
    distorter = create_distorter(rir_files, bg_files)

    # Load sampler
    sampler = load_effected_sampler(datasets, distorter, duration, return_source)

    return sampler


def load_mixed_sampler(datasets, duration, return_source = False):

    # Target duration
    frames = int(duration * config.audio.sample_rate)

    # Load the datasets
    files = []
    for dataset in datasets:
        dataset_files = list(Path(dataset).rglob("*.wav")) + list(Path(dataset).rglob("*.flac"))
        dataset_files = [str(p) for p in dataset_files]
        files += dataset_files

    # Load BG files
    bg_files = []
    for p in Path("./external_datasets/dns-noise").rglob("*.wav"):
        bg_files.append(str(p))

    # Pipelines
    common_pipeline = light_noisy_common_pipeline()
    clean_pipeline = light_noisy_pipeline(bg = bg_files)
    voiced_pipeline = light_noisy_voiced_pipeline(bg = bg_files, voices = files)

    # Do sampling
    def sampler():

        # Pick target
        if random.random() < 0.9:

            # Load audio
            f = random.choice(files)
            audio = load_mono_audio(f, config.audio.sample_rate)
            if audio.shape[0] < frames:
                padding = frames - audio.shape[0]
                padding_left = random.randint(0, padding)
                padding_right = padding - padding_left
                audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)
            else:
                start = random.randint(0, audio.shape[0] - frames)
                audio = audio[start:start + frames]

            # Target is loaded audio
            target = torch.clone(audio)

            # Audio is lightly noisy audio with possible voices
            audio = voiced_pipeline.apply(audio, config.audio.sample_rate)
        else:

            # Empty target
            target = torch.zeros(frames)

            # Light noisy audio without voices
            audio = clean_pipeline.apply(torch.zeros(frames), config.audio.sample_rate)

        # Apply common pipeline (guassian noise and effects are similar for both target and audio)
        r = common_pipeline.resolve()
        audio = common_pipeline.apply(audio, config.audio.sample_rate, resolved = r)
        target = common_pipeline.apply(target, config.audio.sample_rate, resolved = r)

        # Spectogram
        spec = spectogram(target, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        # Spectogram with effect
        spec_with_effect = spectogram(audio, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        if return_source:
            return spec, spec_with_effect, target, audio
        else:
            return spec, spec_with_effect

    return sampler

#
# Loaders
#

def load_distorted_loader(datasets, duration, batch_size, num_workers, return_source = False):

    # Load sampler
    sampler = load_distorted_sampler(datasets, duration, return_source)

    # Load dataset
    class DistortedDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = DistortedDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader

def load_clean_loader(datasets, duration, batch_size, num_workers, return_source = False):

    # Load sampler
    sampler = load_clean_sampler(datasets, duration, return_source)

    # Load dataset
    class DistortedDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = DistortedDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader
    
def load_mixed_loader(datasets, duration, batch_size, num_workers, return_source = False):

    # Load sampler
    sampler = load_mixed_sampler(datasets, duration, return_source = return_source)

    # Load dataset
    class DistortedDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = DistortedDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader