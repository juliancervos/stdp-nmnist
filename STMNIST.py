import re
import os
import os.path
from datetime import datetime

import numpy as np
import torch

from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from eventpython.eventvision import *
import scipy.io
from sklearn.model_selection import train_test_split


class STMNIST(VisionDataset):
    # language=rst
    """`ST-MNIST <https://www.benjamintee.com/stmnist/>`_ Dataset.

    :param root: Root directory of dataset where ``STMNIST/processed/training.pt`` and  ``STMNIST/processed/test.pt``
    exist.
    :param train: If True, creates dataset from ``training.pt``, otherwise from ``test.pt``.
    :param transform: A function/transform that  takes in a tensor and returns a transformed version.
    :param target_transform: A function/transform that takes in the target and transforms it.
    :param download: If true, downloads the dataset from the internet and puts it in root directory. If dataset is
        already downloaded, it is not downloaded again.
    :param dt: Simulation timestep (used for the binning of the data).
    """

    resources = ["https://www.dropbox.com/s/89w329um9rp9qha/STMNIST%20dataset%20NUS%20Tee%20Research%20Group.zip?dl=1"]

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            dt: float = 1.0,
            debug: bool = False,
    ) -> None:
        super(STMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.dt = dt
        self.debug = debug

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        with open(os.path.join(self.processed_folder, data_file), 'rb') as pickled_file:
            self.data, self.targets = torch.load(pickled_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # language=rst
        """
        :param index: Index

        :return: Tuple: (event_image, target) where target is index of the target class.
        """
        ev_img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            ev_img = self.transform(ev_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return ev_img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def training_file(self) -> str:
        return f'training{int(self.dt)}.pt'

    @property
    def test_file(self) -> str:
        return f'test{int(self.dt)}.pt'

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self) -> None:
        # language=rst
        """
        Download the ST-MNIST data if it doesn't exist in processed_folder already.
        """

        if self._check_exists() and not self.debug:
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download files
        dataset_url = self.resources[0]
        dataset_filename = 'STMNIST_dataset.zip'
        download_and_extract_archive(dataset_url, download_root=self.raw_folder, filename=dataset_filename)

        # process and save as torch files
        print('Processing...')

        training_set = self.read_event_image_files(os.path.join(self.raw_folder, 'data_submission'), train=True)
        test_set = self.read_event_image_files(os.path.join(self.raw_folder, 'data_submission'), train=False)

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    def read_event_image_files(self, path: str, train: bool, three_saccades: bool = False) -> Tuple[List[torch.Tensor],
                                                                                                    torch.ByteTensor]:
        # language=rst
        """
        Reads the data (previously downloaded) consisting of event-based images and extract their labels.

         :param path: Path to the data file to load (train or test).

         :return: Tuple (spikes_array, labels) where spikes_array is a list of event_images and labels is a tensor
         with their
         corresponding ground-truth.
         """
        spikes_array, labels = [], []

        reference_matrix = torch.flatten(torch.transpose(torch.arange(0, 100, dtype=torch.int32).view(10, 10), 0, 1))
        image_time = 2000.0

        number_of_timesteps = int(image_time / self.dt)
        bins = np.arange(0.0, image_time, self.dt, dtype=np.int16)

        stmnist_directories = (d for d in os.listdir(path) if d != 'LUT.mat')
        for directory_number in stmnist_directories:
            for file in os.listdir(os.path.join(path, directory_number)):
                event_image = scipy.io.loadmat(os.path.join(path, directory_number, file))['spiketrain']
                event_image = np.delete(event_image, np.where(event_image == -1.0)[1], axis=1)  # only ON events

                event_number = np.where(event_image[:-1] == 1.0)[1]
                unique_event_numbers, unique_event_numbers_counts = np.unique(event_number, return_counts=True)
                corrupted_entries = np.where(unique_event_numbers_counts != 1)
                cleaned_data = np.delete(event_image, corrupted_entries, axis=1)
                # event_count = cleaned_data[:-1].sum(axis=1)
                filter_indexes = np.where(cleaned_data[:-1].sum(axis=1) > 40.0)
                anomalous_entries = np.where(cleaned_data[filter_indexes] > 0.0)[1]
                filtered_data = np.delete(cleaned_data, anomalous_entries, axis=1)

                timestamps_ms = torch.from_numpy(filtered_data[-1] * 1000)
                neuron_indexes = reference_matrix[np.nonzero(np.transpose(filtered_data[:-1]) == 1.0)[1]]

                bin_indexes = torch.tensor(np.digitize(timestamps_ms, bins) - 1, dtype=torch.int32)
                sparse_indexes = torch.stack([bin_indexes, neuron_indexes], dim=0)
                sparse_values = torch.ones((neuron_indexes.size()), dtype=torch.uint8)
                spikes = torch.sparse_coo_tensor(sparse_indexes, sparse_values, (number_of_timesteps, 100))

                spikes_array.append(spikes)
                labels.append(int(directory_number))

        train_array, test_array = train_test_split(spikes_array, test_size=0.2, shuffle=True, random_state=0)

        spikes_array = train_array if train else test_array

        return spikes_array, torch.ByteTensor(labels)
