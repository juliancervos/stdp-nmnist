import os
import os.path
import torch
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from eventpython.eventvision import *


class NMNIST(VisionDataset):
    # language=rst
    """
    `N-MNIST <https://www.garrickorchard.com/datasets/n-mnist>`_ Dataset.

    :param root: Root directory of dataset where ``NMNIST/processed/training.pt`` and  ``NTIDIGITS/processed/test.pt`` exist.
    :param train: If True, creates dataset from ``training.pt``, otherwise from ``test.pt``.
    :param transform: A function/transform that  takes in a tensor and returns a transformed version.
    :param target_transform: A function/transform that takes in the target and transforms it.
    :param download: If true, downloads the dataset from the internet and puts it in root directory. If dataset is
        already downloaded, it is not downloaded again.
    :param dt: Simulation timestep (used for the binning of the data).
    """

    resources = ["https://www.dropbox.com/s/4vghfgan28nt9ih/Train.zip?dl=1",
                 "https://www.dropbox.com/s/c4hbhgo2fevmtww/Test.zip?dl=1"]

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            dt: float = 1.0,
    ) -> None:
        super(NMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
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
        Download the N-MNIST data if it doesn't exist in processed_folder already.
        """

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download files
        trainset_url = self.resources[0]
        trainset_filename = 'NMNIST_Train.zip'
        download_and_extract_archive(trainset_url, download_root=self.raw_folder, filename=trainset_filename)
        testset_url = self.resources[1]
        testset_filename = 'NMNIST_Test.zip'
        download_and_extract_archive(testset_url, download_root=self.raw_folder, filename=testset_filename)

        # Process and save as torch files
        print('Processing...')

        training_set = self.read_event_image_files(os.path.join(self.raw_folder, 'Train'))
        test_set = self.read_event_image_files(os.path.join(self.raw_folder, 'Test'))

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    def read_event_image_files(self, path: str, three_saccades:bool = False) -> Tuple[List[torch.Tensor],
                                                                                   torch.ByteTensor]:
        # language=rst
        """
        Reads the data (previously downloaded) consisting of event-based images and extract their labels.

         :parma path: Path to the data file to load (train or test).
         :param three_saccades: If True, include all 3 saccades from the dataset, otherwise only the first one.

         :return: Tuple (spikes_array, labels) where spikes_array is a list of event_images and labels is a tensor with their
         corresponding ground-truth.
         """
        # Initialise variables and parameters
        spikes_array, labels = [], []
        reference_matrix = torch.transpose(torch.arange(0, 1156, dtype=torch.int32).view(34, 34), 0, 1)
        # TODO: Obtain this as an argument from Visual_network.py
        image_time = 315.0
        saccade_time = image_time/3
        number_of_timesteps = int(saccade_time / self.dt)

        # Compute number of bins
        bins = np.arange(0.0, saccade_time, self.dt, dtype=np.int16)

        # Iterate over all files and directories
        for directory_number in os.listdir(path):
            for file in os.listdir(os.path.join(path, directory_number)):
                # Read data from file
                event_image = read_dataset(os.path.join(path, directory_number, file)).data

                # Extract timestamps in miliseconds for ON events
                pattern_ts = event_image.ts[event_image.p] / 1000

                # Split events into saccades
                saccades_indexes = []
                saccades_indexes.append(np.argwhere(pattern_ts[pattern_ts < saccade_time]).squeeze())
                if three_saccades:
                    saccades_indexes.append(np.argwhere(pattern_ts[(pattern_ts > saccade_time) &
                                                                   (pattern_ts < saccade_time*2)]).squeeze())
                    saccades_indexes.append(np.argwhere(pattern_ts[pattern_ts > saccade_time*2]).squeeze())

                for saccade in saccades_indexes:
                    # Get events addresses and timestamps
                    neuron_indexes = reference_matrix[event_image.x[saccade].astype(np.int32),
                                                      event_image.y[saccade].astype(np.int32)].clone().detach()
                    timestamps_ms = pattern_ts[saccade]

                    # Bin events according to binning windows and timestamps
                    bin_indexes = torch.tensor(np.digitize(timestamps_ms, bins) - 1, dtype=torch.int32)

                    # Construct sparse matrix
                    sparse_indexes = torch.stack([bin_indexes, neuron_indexes], dim=0)
                    sparse_values = torch.ones((neuron_indexes.size()), dtype=torch.uint8)
                    spikes = torch.sparse_coo_tensor(sparse_indexes, sparse_values, (number_of_timesteps, 1156))

                    # Append sparse matrix and label
                    spikes_array.append(spikes)
                    labels.append(int(directory_number))

        return spikes_array, torch.ByteTensor(labels)


class SparseToDense(object):
    # language=rst
    """
    Convert the sparse tensor given in sample to a dense tensor.
    """

    def __call__(self, sample):
        dense_spikes = sample.to_dense()
        # TODO: Try substituting torch.ones_like() with "1" (or a scalar, should work)
        return torch.where(dense_spikes > 1, torch.ones_like(dense_spikes), dense_spikes)

