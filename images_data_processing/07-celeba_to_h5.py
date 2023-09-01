import os
import cv2
import torch
import datetime
import numpy as np
import torchvision
import face_alignment
from typing import Tuple, TextIO

from PIL import Image, ImageOps
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import proj_path
from utils.utils import store_many_hdf5
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_box_by_facial_landmarks(landmarks: np.ndarray, additive_coefficient: float = 0.1) -> list:
    """
    Calculate face bounding box based on facial landmarks.

    Args:
        landmarks (np.ndarray): Facial landmarks points in int32 datatype and XY format.
        additive_coefficient (float, optional): Value of additive face area on box. Default is 0.1.

    Returns:
        list: Bounding box data in XYXY format [x_min, y_min, x_max, y_max].
    """
    # Calculate minimum and maximum x, y coordinates
    x0 = landmarks[..., 0].min()
    y0 = landmarks[..., 1].min()
    x1 = landmarks[..., 0].max()
    y1 = landmarks[..., 1].max()

    # Calculate delta x and delta y based on the additive_coefficient
    dx = int((x1 - x0) * additive_coefficient)
    dy = int((y1 - y0) * additive_coefficient * 2)

    # Calculate bounding box coordinates [x_min, y_min, x_max, y_max]
    box = [
        x0 - dx // 2, y0 - dy,
        x1 + dx // 2, y1 + dy // 3
    ]
    return box


def create_square_crop_by_detection(frame: np.ndarray, box: list) -> np.ndarray:
    """
    Create a square crop from the input image based on the given detection box.

    Args:
        frame (np.ndarray): RGB image in np.uint8 format.
        box (list): List specifying the detection box in the format [x1, y1, x2, y2].

    Returns:
        np.ndarray: Cropped image with a square shape.
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w // 2
    cy = box[1] + h // 2
    radius = max(w, h) // 2
    exist_box = []
    pads = []

    # Calculate valid y range (top)
    if cy - radius >= 0:
        exist_box.append(cy - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cy - radius))

    # Calculate valid y range (bottom)
    if cy + radius >= frame.shape[0]:
        exist_box.append(frame.shape[0] - 1)
        pads.append(cy + radius - frame.shape[0] + 1)
    else:
        exist_box.append(cy + radius)
        pads.append(0)

    # Calculate valid x range (left)
    if cx - radius >= 0:
        exist_box.append(cx - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cx - radius))

    # Calculate valid x range (right)
    if cx + radius >= frame.shape[1]:
        exist_box.append(frame.shape[1] - 1)
        pads.append(cx + radius - frame.shape[1] + 1)
    else:
        exist_box.append(cx + radius)
        pads.append(0)

    # Crop the valid region from the frame
    exist_crop = frame[
        exist_box[0]:exist_box[1],
        exist_box[2]:exist_box[3]
    ]
    
    # Pad the cropped image to make it square
    croped = np.pad(
        exist_crop,
        (
            (pads[0], pads[1]),
            (pads[2], pads[3]),
            (0, 0)
        ),
        'constant'
    )
    return croped


def transform_picture(img_sample: np.array, landmarks: np.array, resize: torch, greyscale: bool = True):
    """
    Crop an image based on landmarks and transform it into grayscale.

    Args:
        img_sample (np.ndarray): Image to process.
        landmarks (np.ndarray or None): Landmarks of the face or None.
        resize (function): Function to resize the image.
        greyscale (bool): Flag to determine whether to convert to grayscale.

    Returns:
        np.ndarray: Transformed image.
    """
    # If landmarks are detected, crop the picture
    if landmarks is not None:
        landmarks = np.floor(landmarks[0]).astype(np.int32)
        face_box = get_box_by_facial_landmarks(landmarks)
        img_sample = create_square_crop_by_detection(img_sample, face_box)

    # Convert to PIL Image
    PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')

    # Convert to grayscale if greyscale flag is True
    if greyscale:
        PIL_image = ImageOps.grayscale(PIL_image)

    # Resize the image using the provided resize function
    sample = np.asarray(resize(PIL_image), dtype=np.uint8)
    return sample

def make_array(pictures_dir: str, identity_file: TextIO, device: str, size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create arrays of images and corresponding labels.

    Args:
        pictures_dir (str): Path to the image folder.
        identity_file: File containing image identities.
        device: Device to use for face alignment.
        size (tuple): Desired size for the images.
        logger: Logger instance for logging messages.

    Returns:
        np.ndarray: Array of images.
        np.ndarray: Array of labels.
    """

    logger.info(f"Creating arrays for folder: {pictures_dir}")

    # Concatenate array of images
    img_array = []
    label_array = []

    # Initialize FaceAlignment
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device=device
    )

    # Resize images to specified size
    resize = torchvision.transforms.Resize(size)

    for line in identity_file.readlines():
        im_path = os.path.join(pictures_dir, line[0:10])

        if not os.path.isfile(im_path):
            logger.warning(f"Image not found: {im_path}")
            continue

        img_sample = cv2.imread(im_path, cv2.COLOR_BGR2RGB)
        if img_sample is None:
            logger.warning(f"Image not found: {im_path}")
            continue
        # Get facial landmarks
        landmarks = fa.get_landmarks_from_image(img_sample)

        # Transform the image
        sample = transform_picture(img_sample, landmarks, resize)

        # Extract label
        label = np.array(int(line[11:-1]))

        # Append image and label to arrays
        img_array.append(sample)
        label_array.append(label)

    img_array = np.asarray(img_array)
    label_array = np.asarray(label_array)

    logger.info(f"Label array shape: {label_array.shape}")
    logger.info(f"Image array shape: {img_array.shape}")

    return img_array, label_array

if __name__ == '__main__':
    begin_time = datetime.datetime.now()

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    for type_data in ["with", "without"]:
        for folder in ["valid", "test", "train"]:
            dir_txt = os.path.join(proj_path, "images_data_processing", "files", f"identity_CelebA_{folder}_{type_data}_meg_stimuli.txt")
            identity_file = open(dir_txt, "r")
            img_array, label_array = make_array(f"{folder}_{type_data}_meg_stimuli", identity_file, device)
            store_many_hdf5(img_array, label_array, f"{folder}_{type_data}_meg_stimuli")
            logger.info(f"{folder}, {type_data} meg stimuli processed and stored to HDF5.")

    logger.info(f"Total time taken: {datetime.datetime.now() - begin_time}")
