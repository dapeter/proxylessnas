# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchaudio

from data_providers.base_provider import *


def loader(path):
    waveform, sample_rate = torchaudio.load(path, normalization=True)
    padded_waveform = torch.zeros([1, sample_rate])
    padded_waveform[0, 0:len(waveform[0])] = waveform[0]

    params = {
        "channel": 0,
        "dither": 0.0,
        "window_type": "hanning",
        "frame_length": 40,
        "frame_shift": 20,
        "remove_dc_offset": False,
        "round_to_power_of_two": False,
        "sample_frequency": sample_rate,
    }

    mfcc = torchaudio.compliance.kaldi.mfcc(padded_waveform, **params)
    return mfcc.unsqueeze(0)


class SpeechCommandsDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):
        self._save_path = save_path

        train_dataset = datasets.DatasetFolder(self.save_path + "train/", loader, extensions=('.wav',))
        validation_dataset = datasets.DatasetFolder(self.save_path + "val/", loader, extensions=('.wav',))
        test_dataset = datasets.DatasetFolder(self.save_path + "test/", loader, extensions=('.wav',))

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
        # return 1, 40, 49  # C, H, W : should be
        return 1, 49, 13
        #return 1, 40, 51 # C, H, W

    # TODO add silence
    @property
    def n_classes(self):
        return 10 #11 #12

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = 'dataset/speech_commands/'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download speech commands')
