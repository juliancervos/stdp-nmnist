import os
import os.path
import h5py
import torch
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, download_url
from eventpython.eventvision import *


class NTIDIGITS(VisionDataset):
    # language=rst
    """
    `N-TIDIGITS <https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit#>`_ Dataset.

    :param root: Root directory of dataset where ``NTIDIGITS/processed/training.pt`` and  ``NTIDIGITS/processed/test.pt`` exist.
    :param train: If True, creates dataset from ``training.pt``, otherwise from ``test.pt``.
    :param transform: A function/transform that  takes in a tensor and returns a transformed version.
    :param target_transform: A function/transform that takes in the target and transforms it.
    :param download: If true, downloads the dataset from the internet and puts it in root directory. If dataset is
        already downloaded, it is not downloaded again.
    :param dt: Simulation timestep (used for the binning of the data).
    """

    resource = "https://dl.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=1"

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            dt: float = 1.0,
    ) -> None:
        super(NTIDIGITS, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.dt = dt

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

        :return: Tuple: (utterance, target) where target is index of the target class.
        """
        utterance, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            utterance = self.transform(utterance)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return utterance, target

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
        Download the N-TIDIGITS data if it doesn't exist in processed_folder already.
        """

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download files
        download_url(self.resource, root=self.raw_folder, filename='n-tidigits.hdf5')

        # Process and save as torch files
        print('Processing...')

        training_set = self.read_event_audio_files(os.path.join(self.raw_folder, 'n-tidigits.hdf5'), True)
        test_set = self.read_event_audio_files(os.path.join(self.raw_folder, 'n-tidigits.hdf5'), False)

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    def read_event_audio_files(self, path: str, train: bool) -> Tuple[List[torch.Tensor], torch.ByteTensor]:
        # language=rst
        """
        Reads the data previously downloaded into tensors, binning the events according to the simulation timestep in the
        process.

         :param path: Path to the data file to load (train or test).
         :param train: If True, read training data.

         :return: Tuple (spikes_array, labels) where spikes_array is a list of utterances and labels is a tensor with their
         corresponding ground-truth.
         """
        # Initialise variables and parameters
        ntidigits = h5py.File(path, 'r')
        dataset_group = 'train' if train else 'test'
        spikes_array, labels = [], []
        image_time = 1000.0
        number_of_timesteps = int(image_time / self.dt)

        # Compute number of bins
        bins = np.arange(0.0, image_time, self.dt, dtype=np.int16)

        # Iterate over all single-digit utterances (excluding 'oh' class)
        for sample_key in ntidigits[dataset_group + '_addresses']:
            if len(sample_key.rpartition('-')[2]) == 1 \
                    and (sample_key.rpartition('-')[2].isdigit() or sample_key.rpartition('-')[2] == 'z'):
                # Read data from file (events addresses and timestamps and class label)
                neuron_indexes = torch.from_numpy(ntidigits[dataset_group + '_addresses'][sample_key][...])
                timestamps_ms = torch.from_numpy(ntidigits[dataset_group + '_timestamps'][sample_key][...]) * 1000
                label = sample_key.rpartition('-')[2]

                # Bin events according to binning windows and timestamps
                bin_indexes = torch.tensor(np.digitize(timestamps_ms, bins) - 1, dtype=torch.int32)

                # Construct sparse matrix
                sparse_indexes = torch.stack([bin_indexes, neuron_indexes], dim=0)
                sparse_values = torch.ones((neuron_indexes.size()), dtype=torch.uint8)
                spikes = torch.sparse_coo_tensor(sparse_indexes, sparse_values, (number_of_timesteps, 64)).to_dense()

                # Append sparse matrix and label
                spikes_array.append(torch.where(spikes > 1, torch.ones_like(spikes), spikes))
                labels.append(0 if label == 'z' else int(label))

        return spikes_array, torch.ByteTensor(labels)
