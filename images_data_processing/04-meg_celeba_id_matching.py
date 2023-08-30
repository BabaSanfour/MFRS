import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import study_path, proj_path

# Initialize dlib components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(proj_path, "utils", "shape_predictor_68_face_landmarks.dat"))
face_encoder = dlib.face_recognition_model_v1(os.path.join(proj_path, "dlib_face_recognition_resnet_model_v1.dat"))

# Utilities
def face_distance(face_encodings: np.array, face_to_compare: np.array) -> float:
    """
    Calculate the Euclidean distance between two face encodings.

    Args:
    ----------
    face_encodings : np.ndarray
        The known face encodings to compare against.
    face_to_compare : np.ndarray
        The face encoding to compare.

    Returns:
    -------
    distance : float
        The Euclidean distance between the face encodings.
    """
    return np.linalg.norm(face_encodings - face_to_compare)

def compare_faces(known_face_encodings: list, face_encoding_to_check: list, tolerance: int = 0.6) -> list:
    """
    Compare a face encoding to a list of known face encodings.

    Args:
    ----------
    known_face_encodings : list
        List of known face encodings.
    face_encoding_to_check : np.ndarray
        Face encoding to compare.
    tolerance : float, optional
        Tolerance threshold for considering a match, by default 0.6.

    Returns:
    -------
    comparison_result : list
        List containing the comparison result (True or False) and the distance.
    """
    torf = (face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    dis = np.round(face_distance(known_face_encodings, face_encoding_to_check), 2)
    return [torf, dis]

if __name__ == '__main__':
    # Load image paths from stimuli directory
    stimuli_folder = os.path.join(study_path, 'web_scrapping_stimuli/')
    stimuli_paths = sorted([os.path.join(stimuli_folder, sname) for sname in os.listdir(stimuli_folder)])

    # Load processed data from all_data folder
    all_data_folder = os.path.join(study_path, 'all_data/')
    all_data_paths = sorted([os.path.join(all_data_folder, sname) for sname in os.listdir(all_data_folder)])

    # Check if embeddings file exists
    embeddings_path = "celebA_embeddings.npy"
    if not os.path.exists(embeddings_path):
        embeddings = []
        embeddings_generator = tqdm(all_data_paths)
        for picture_path in embeddings_generator:
            img = cv2.imread(picture_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_face = detector(img_gray)[0]

            landmarks = predictor(img_gray, img_face)
            face_embedding = np.array(face_encoder.compute_face_descriptor(img_face, landmarks, num_jitters=1))
            embeddings.append(face_embedding)

        np.save(embeddings_path, np.array(embeddings))
    else:
        embeddings = list(np.load(embeddings_path))

    # Loop through stimuli images
    selected = []
    for id in range(150):
        img = cv2.imread(stimuli_paths[id])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector(gray)[0]
        landmarks = predictor(gray, face)
        main_face_embedding = np.array(face_encoder.compute_face_descriptor(img, landmarks, num_jitters=1))

        list_comp = []
        for i, list_of_face_embedding in enumerate(embeddings):
            if len(list_of_face_embedding) == 0:
                list_comp.append([i, -1])
                continue
            listof = compare_faces(list_of_face_embedding[0], main_face_embedding, tolerance=0.6)
            list_comp.append(listof)

        min_score, max_score = 0.0, 0.5
        for i, item in enumerate(list_comp):
            if type(item[0]) == int:
                continue
            if item[1] >= min_score and item[1] < max_score:
                selected.append(i)

        # Plot selected images
        fig, axes = plt.subplots(len(selected), 1, figsize=(15, len(selected) * 5))
        for i, selected_idx in enumerate(selected):
            im = plt.imread(all_data_paths[selected_idx])
            axes[i].imshow(im)

    # Define and save the mapping dictionary: Done after double verification.
    matched = {
    4: 8284, 5: 5923, 8: 8134, 16: 1182, 19: 6487,
    29: 7880, 36: 2178, 102: 7515, 105: 948, 107: 2090, 109: 4403, 
    123: 125,  125: 4499, 130: 5820, 136: 4698, 138: 4657, 53: 8367,
    57: 356, 62: 3615, 65: 1647, 68: 4762, 69: 7492, 80: 2267, 81: 4990,
    }