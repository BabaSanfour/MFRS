import os
import pickle
import logging
import pandas as pd
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from utils.config import study_path

# Mapping of matched IDs to their corresponding indices
matched_ids_mapping = {
    4: 8284, 5: 5923, 8: 8134, 16: 1182, 19: 6487,
    29: 7880, 36: 2178, 102: 7515, 105: 948, 107: 2090, 109: 4403, 
    123: 125,  125: 4499, 130: 5820, 136: 4698, 138: 4657, 53: 8367,
    57: 356, 62: 3615, 65: 1647, 68: 4762, 69: 7492, 80: 2267, 81: 4990,
}

if __name__=="__main__":
    # Load old CelebA labels
    old_celebA_labels = pd.read_csv('identity_CelebA.txt', sep=" ", header=None)
    old_celebA_labels.columns = ["name", "label"]

    # Get the list of stimuli names
    stimuli_names = sorted(os.listdir(os.path.join(study_path, 'web_scrapping_stimuli/')))

    # Update labels for matched IDs
    for key, id in matched_ids_mapping.items():
        occurrences = old_celebA_labels[old_celebA_labels["label"] == id].count()[0]
        stimuli = f'f{key:03d}'
        matched_ids_mapping[stimuli] = [id, occurrences]

    counter_images = len(old_celebA_labels)

    # Process matched IDs images
    for stimuli, values in matched_ids_mapping.items():
        id, occurrences = values
        scrapped_data_folder = os.path.join(study_path, 'web_scrapping_stimuli/', stimuli)
        scrapped_data_pictures = sorted(
            [os.path.join(scrapped_data_folder, sname) for sname in os.listdir(scrapped_data_folder)]
        )
        for image in scrapped_data_pictures[:30 - occurrences]:
            im = Image.open(image)
            im = im.convert("RGB")
            im.save(os.path.join(study_path, f'img_align_celeba/img_align_celeba/{counter_images}.jpg'))
            list_row = [f'{counter_images}.jpg', id]
            old_celebA_labels.loc[len(old_celebA_labels)] = list_row
            counter_images += 1

    # Process remaining stimuli
    stimuli_ids = [values[0] for values in matched_ids_mapping.values()]
    counter_images = len(old_celebA_labels)
    new_id = max(old_celebA_labels["label"]) + 1
    for stimuli in stimuli_names:
        if stimuli in matched_ids_mapping:
            continue
        scrapped_data_folder = os.path.join(study_path, 'web_scrapping_stimuli/', stimuli)
        scrapped_data_pictures = sorted(
            [os.path.join(scrapped_data_folder, sname) for sname in os.listdir(scrapped_data_folder)]
        )
        for image in scrapped_data_pictures[:30]:
            im = Image.open(image)
            im = im.convert("RGB")
            im.save(os.path.join(study_path, f'img_align_celeba/img_align_celeba/{counter_images}.jpg'))
            list_row = [f'{counter_images}.jpg', new_id]
            old_celebA_labels.loc[len(old_celebA_labels)] = list_row
            counter_images += 1
        stimuli_ids.append(new_id)
        matched_ids_mapping[stimuli] = [new_id, 30]
        new_id += 1

    # Save new file and mapping dict
    old_celebA_labels.to_csv('/files/identity_CelebA_with_meg_stimuli.txt', header=None, index=None, sep=' ', mode='a')
    logger.info(f"Mapping dictionary saved to identity_CelebA_with_meg_stimuli.txt")

    with open('/files/mapping_dict.pickle', 'wb') as pickle_file:
        pickle.dump(matched_ids_mapping, pickle_file)
    logger.info(f"Mapping dictionary saved to mapping_dict.pickle")


