import numpy as np
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Optional


def npy_loader(path):
    sample = np.load(path)
    return sample


NPY_EXTENSIONS = (".npy",)


class ECGDatasetFolder(DatasetFolder):
    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = npy_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None, ):

        super(ECGDatasetFolder, self).__init__(root, loader, NPY_EXTENSIONS if is_valid_file is None else None,
                                               transform=transform,
                                               target_transform=target_transform,
                                               is_valid_file=is_valid_file)
