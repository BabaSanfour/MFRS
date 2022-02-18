"""
    For each txt file:
    - Create a HDF5 file with pictures and their IDs.
    For each picture:
    - Get face bounding box by landmarks points.
    - Crop face to include only face, forehead, chin and part of the hair.
    ---------------
    Input Files:
    ---------------
    read files in files/txt_files:       24 txt file with pictures name and id
    ---------------
    Output Files:
    ---------------
    24 HDF5 files in data/MFRS_data/HDF5_files: 3 HDF5 files for each data repartition set: Train, valid and test classes
    Parameters:
    images       images array, (N, 224, 224, 1) to be stored
    labels       labels array, (N, ) to be stored

"""

import os
import cv2
import h5py
import datetime
import numpy as np
import torchvision
import face_alignment
from PIL import Image, ImageOps

dic = {25:1000, 12:1000,
        27:500, 13:500,
        28:300, 14:300,
        29:150, 15:150}

root_dir = "/home/hamza97/"


def get_box_by_facial_landmarks(
        landmarks: np.ndarray,
        additive_coefficient: float = 0.2) -> list:
    """
    Configure face bounding box by landmarks points
    Args:
        landmarks: landmarks points in int32 datatype and XY format
        additive_coefficient: value of additive face area on box

    Returns:
        List with bounding box data in XYXY format
    """
    x0 = landmarks[..., 0].min()
    y0 = landmarks[..., 1].min()

    x1 = landmarks[..., 0].max()
    y1 = landmarks[..., 1].max()

    dx = int((x1 - x0) * additive_coefficient)
    dy = int((y1 - y0) * additive_coefficient * 2)

    return [
        x0 - dx // 2, int(y0 - dy * 3),
        x1 + dx // 2, y1 + dy // 2]

def create_square_crop_by_detection(frame: np.ndarray, box: list) -> np.ndarray:
    """
    Rebuild detection box to square shape
    Args:
        frame: rgb image in np.uint8 format
        box: list with follow structure: [x1, y1, x2, y2]
    Returns:
        Image crop by box with square shape
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w // 2
    cy = box[1] + h // 2
    radius = max(w, h) // 2
    exist_box = []
    pads = []

    # y top
    if cy - radius >= 0:
        exist_box.append(cy - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cy - radius))

    # y bottom
    if cy + radius >= frame.shape[0]:
        exist_box.append(frame.shape[0] - 1)
        pads.append(cy + radius - frame.shape[0] + 1)
    else:
        exist_box.append(cy + radius)
        pads.append(0)

    # x left
    if cx - radius >= 0:
        exist_box.append(cx - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cx - radius))

    # x right
    if cx + radius >= frame.shape[1]:
        exist_box.append(frame.shape[1] - 1)
        pads.append(cx + radius - frame.shape[1] + 1)
    else:
        exist_box.append(cx + radius)
        pads.append(0)

    exist_crop = frame[
                 exist_box[0]:exist_box[1],
                 exist_box[2]:exist_box[3]
                 ]
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

def store_many_hdf5(images, labels, folder):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 224, 224, 1) to be stored
        labels       labels array, (N, ) to be stored
    """
    hdf5_dir = root_dir+"scratch/data/MFRS_data/hdf5/"
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    # Create a new HDF5 file
    file = h5py.File("%s%s.h5"%(hdf5_dir,folder), "w")
    print("{} h5 file created".format(folder))

    # Create a dataset in the file
    dataset = file.create_dataset("images", np.shape(images), data=images) #h5py.h5t.STD_U8BE,
    metaset = file.create_dataset("meta", np.shape(labels), data=labels)
    file.close()
    print("{} h5 is ready".format(folder))


def transform_picture(img_sample, landmarks, resize):
    """ Crop an image and transform it into grayscale.
        Parameters:
        ---------------
        img_sample       image to process
        landmarks        landmarks of the face
    """
    # if we detected landmarks we crop the picture
    if landmarks != None:
        landmarks = np.floor(landmarks[0]).astype(np.int32)
        face_box = get_box_by_facial_landmarks(landmarks)
        img_sample = create_square_crop_by_detection(img_sample, face_box)

    # Otherwise we transform it directly to greyscale
    PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
    PIL_image = ImageOps.grayscale(PIL_image)
    sample = np.asarray(resize(PIL_image), dtype=np.uint8)
    return sample


def make_array(data_folder, identity_file):
    pictures_dir = root_dir+"scratch/data/MFRS_data/"+data_folder+"/"


    # Concatenate array of images
    img_array = []
    label_array = []
    # initilize FaceAlignment
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device='cpu'
        )

    # resize images to 224 224
    resize = torchvision.transforms.Resize((224, 224))

    for  line in identity_file.readlines():
        # read the image        print(img_sample)

        img_sample = cv2.imread(os.path.join(pictures_dir, line[0:10]), cv2.COLOR_BGR2RGB)
        # Get facial landmarks
        landmarks = fa.get_landmarks_from_image(img_sample)
        # transform the image
        sample=transform_picture(img_sample, landmarks, resize)
        # extract label
        label = np.array(int(line[11:-1]))

        # append image and label to image and labels list
        img_array.append(sample)
        label_array.append(label)

    # print label and image array shapes
    print(np.asarray(label_array).shape)
    print(np.asarray(img_array).shape)
    # return image and label arrays
    return np.asarray(img_array), np.asarray(label_array)

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    for key, length in dic.items():
        for folder in ["train_%s_%s"%(length, key),"test_%s_%s"%(length, key),"valid_%s_%s"%(length, key)] :
            dir_txt = root_dir + "MFRS/files/txt_files/identity_CelebA_%s.txt"%folder
            identity_file = open(dir_txt, "r")
            img_array, label_array = make_array(folder, identity_file)
            store_many_hdf5(img_array,label_array, folder)
    print(datetime.datetime.now()-begin_time)
