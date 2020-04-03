# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchaudio

from data_providers.base_provider import *

import re
import hashlib

# TODO unify functions
def is_train(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension == '.wav':
        return which_set(path) == 'training'
    return False

def is_val(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension == '.wav':
        return which_set(path) == 'validation'
    return False

def is_test(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension == '.wav':
        return which_set(path) == 'testing'
    return False

# TODO optimize
def loader(path):
    waveform, sample_rate = torchaudio.load(path, normalization=True)
    padded_waveform = torch.zeros([1, 16000])
    padded_waveform[0, 0:len(waveform[0])] = waveform[0]
    mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=40, win_length=640, n_fft=640)(padded_waveform)
    return mel_specgram

#TODO: dont fix percentages
def which_set(filename, validation_percentage=10, testing_percentage=10):
    MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M

    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

class SpeechCommandsDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):
        self._save_path = save_path

        train_dataset = datasets.DatasetFolder(self.save_path, loader, is_valid_file=is_train)
        validation_dataset = datasets.DatasetFolder(self.save_path, loader, is_valid_file=is_val)
        test_dataset = datasets.DatasetFolder(self.save_path, loader, is_valid_file=is_test)

        self.train = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size,
            num_workers=n_worker, pin_memory=True,
        )
        self.valid = torch.utils.data.DataLoader(
            validation_dataset, batch_size=train_batch_size,
            num_workers=n_worker, pin_memory=True,
        )
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size,
            num_workers=n_worker, pin_memory=True,
        )

    @staticmethod
    def name():
        return 'speech_commands'

    # TODO Width different because of padding, change later
    @property
    def data_shape(self):
        # return 1, 40, 49  # C, H, W : should be
        return 1, 40, 51 # C, H, W

    # TODO add silence
    @property
    def n_classes(self):
        return 11 #12

    # TODO change save path
    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/david/git/proxylessnas/.dataset/speech_commands/speech_commands_v0.01'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download speech commands')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')

    def build_train_transform(self, distort_color, resize_scale):
        return None
