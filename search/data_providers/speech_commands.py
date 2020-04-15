# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision
from torchvision.datasets.vision import VisionDataset
import torchaudio
import os
import os.path
import random
import numpy as np

from data_providers.base_provider import *


class SpeechCommandsDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=100, test_batch_size=100, valid_size=None,
                 n_worker=32, resize_scale=None, distort_color=None):
        self._save_path = save_path

        train_dataset = SpeechCommandsFolder(self.save_path + "train", augment=True)
        validation_dataset = SpeechCommandsFolder(self.save_path + "val", augment=False)
        test_dataset = SpeechCommandsFolder(self.save_path + "test", augment=False)

        self.train = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size,
            num_workers=n_worker, pin_memory=True, shuffle=True
        )
        self.valid = torch.utils.data.DataLoader(
            validation_dataset, batch_size=train_batch_size,
            num_workers=n_worker, pin_memory=True, shuffle=True
        )
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size,
            num_workers=n_worker, pin_memory=True,
        )

    @staticmethod
    def name():
        return 'speech_commands'

    # TODO change width
    @property
    def data_shape(self):
        return 1, 40, 51  # C, H, W

    # TODO add silence
    @property
    def n_classes(self):
        return 12

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = 'dataset/speech_commands/'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download speech commands')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if has_file_allowed_extension(path, extensions):
                    item = (path, class_to_idx[target])
                    images.append(item)

    targets = [i[1] for i in images]
    unknown_indices = [i for i in range(len(targets)) if class_to_idx['unknown'] == targets[i]]
    silence_indices = [i for i in range(len(targets)) if class_to_idx['silence'] == targets[i]]

    all_indices = set(range(len(targets)))
    n_keywords = len(all_indices - set(unknown_indices) - set(silence_indices))
    n_unknowns = n_keywords // 10
    n_silence = n_keywords // 10

    unused_indices = random.sample(unknown_indices, len(unknown_indices) - n_unknowns)
    for i in sorted(unused_indices, reverse=True):
        del images[i]

    for i in range(n_silence - len(silence_indices)):
        silence_index = random.choice(silence_indices)
        images.append(images[silence_index])

    return images


def load_bg_data(dir):
    background_data = []
    background_folder = os.path.join(dir, '_background_noise_')
    for root, _, fnames in os.walk(background_folder):
        for fname in fnames:
            path = os.path.join(root, fname)
            waveform, sample_rate = torchaudio.load(path)
            background_data.append(waveform)

    return background_data


class SpeechCommandsFolder(torch.utils.data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, augment=False):
        self.root = root
        self.augment = augment
        self.extensions = ('.wav',)
        self.sample_rate = 16000
        self.classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        samples = make_dataset(self.root, self.class_to_idx, self.extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.samples = samples
        targets = [s[1] for s in samples]
        self.targets = targets
        self.background_data = load_bg_data(root)

        self.unknown_indices = [i for i in range(len(targets)) if self.class_to_idx['unknown'] == targets[i]]
        self.silence_indices = [i for i in range(len(targets)) if self.class_to_idx['silence'] == targets[i]]
        all_indices = set(range(len(targets)))
        self.keyword_indices = list(all_indices - set(self.unknown_indices) - set(self.silence_indices))
        self.keyword_indices.sort()

        self.n_samples = len(all_indices)

    def extract_features(self, sample):
        n_mfcc = 10
        melkwargs = {
            "win_length": 640,
            "hop_length": 320,
            "n_fft": 640
        }

        mfcc = torchaudio.transforms.MFCC(self.sample_rate, n_mfcc, melkwargs=melkwargs)(sample)

        return mfcc

    def loader(self, path):
        waveform, sample_rate = torchaudio.load(path)
        assert self.sample_rate == sample_rate
        n_samples = waveform.shape[1]

        if n_samples == sample_rate:
            return waveform
        elif n_samples < sample_rate:
            padded_waveform = torch.zeros([1, sample_rate])
            padded_waveform[0, 0:n_samples] = waveform[0]
            return padded_waveform
        elif n_samples > sample_rate:
            raise (RuntimeError("File {} has more than {} samples.".format(path, sample_rate)))

    def transform(self, sample):
        time_shift_ms = 100
        background_frequency = 0.8
        background_volume_range = 0.1

        time_shift_samples = np.random.randint(-time_shift_ms, time_shift_ms) * self.sample_rate // 1000
        shifted_sample = torch.zeros_like(sample)
        sample_len = sample.shape[1]
        if time_shift_samples > 0:
            shifted_sample[:, time_shift_samples:] = sample[:, :sample_len-time_shift_samples]
        elif time_shift_samples < 0:
            time_shift_samples = abs(time_shift_samples)
            shifted_sample[:, :sample_len-time_shift_samples] = sample[:, time_shift_samples:]
        else:
            shifted_sample = sample

        if self.background_data:
            background_sample = random.choice(self.background_data)
            bg_sample_len = background_sample.shape[1]
            background_offset = np.random.randint(0, bg_sample_len - sample_len)
            background_clipped = background_sample[:, background_offset:(background_offset + sample_len)]
            if np.random.uniform(0, 1) < background_frequency:
                background_volume = np.random.uniform(0, background_volume_range)
            else:
                background_volume = 0

            transformed_sample = shifted_sample * (1-background_volume) + background_clipped * background_volume
        else:
            transformed_sample = shifted_sample

        return transformed_sample

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.augment:
            sample = self.transform(sample)
        sample = self.extract_features(sample)

        return sample, target

    def __len__(self):
        return self.n_samples
