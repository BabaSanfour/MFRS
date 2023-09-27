import os
import h5py
import random
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path

def split(a: int, b: int , c: int) -> int:
    """
    Give a random class for each picture based on the input values.

    Parameters:
    ----------
    a : int
        Value of class A.
    b : int
        Value of class B.
    c : int
        Value of class C.

    Returns:
    -------
    int
        Randomly assigned class (0, 1, or 2).
    """
    if a < 0 or b < 0 or c < 0 or (a == 0 and b == 0 and c == 0):
        raise ValueError('Invalid input values')
    
    if a == 0 and b == 0:
        return 2
    elif a == 0 and c == 0:
        return 1
    elif b == 0 and c == 0:
        return 0
    elif b == 0:
        return 2 * random.randrange(2)
    elif c == 0:
        return random.randrange(2)
    elif a == 0:
        return random.randrange(2) + 1
    else:
        return random.randrange(3)


def store_many_hdf5(images: np.ndarray, labels: np.ndarray, file_name: str) -> None:
    """
    Stores an array of images and labels to HDF5.

    Args:
    ----------
    images : np.ndarray
        Images array of shape (N, Width, Height, Number of channels).
    labels : np.ndarray
        Labels array of shape (N, ).
    file_name: str
        file name for HDF5 storage.
    """
    hdf5_dir = os.path.join(study_path, "hdf5")
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    
    # Create a new HDF5 file
    file = h5py.File(os.path.join(hdf5_dir, f"{file_name}.h5"), "w")

    # Create a dataset in the file
    dataset = file.create_dataset("images", np.shape(images), data=images)
    metaset = file.create_dataset("meta", np.shape(labels), data=labels)
    file.close()