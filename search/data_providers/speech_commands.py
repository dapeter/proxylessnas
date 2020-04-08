# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import pickle

from data_providers.base_provider import *


class SpeechCommandsDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):
        self._save_path = save_path

        with open(self.save_path + "speech_commands.p", "rb") as p_file:
            data, class_names = pickle.load(p_file)

        x_train, y_train = data["train"]
        x_val, y_val = data["val"]
        x_test, y_test = data["test"]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        validation_dataset = torch.utils.data.TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

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
        return 1, 40, 51 # C, H, W

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
