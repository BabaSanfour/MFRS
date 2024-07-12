import os
import mne
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rdm import RDM
from src.noise_ceiling import noise_ceiling
from utils.arg_parser import get_similarity_parser
from utils.config import rdms_folder, noise_ceiling_folder, subjects_dir


if __name__ == '__main__':

    parser = get_similarity_parser()
    args = parser.parse_args()
    rdm_instance = RDM(428)

    slicing_indices_meg = {
        "fam": (slice(None), slice(None), slice(None), slice(130), slice(130)),
        "unfamiliar": (slice(None), slice(None), slice(None), slice(130, 280), slice(130, 280)),
        "scrambled": (slice(None), slice(None), slice(None), slice(280, None), slice(280, None))
    }
    if args.brain_analysis_type == "avg":
        brain_activity = []
        for subject in range(1, 17):
            logger.info(f"Loading brain activity for subject {subject:02d}...")
            sub_rdm_path = os.path.join(rdms_folder, f"sub-{subject:02d}", "avg_rdm", f"sub-{subject:02d}_{args.meg_picks}_{args.brain_analysis_type}_rdm_{args.time_segment}_{args.time_window}.npy")
            if not os.path.exists(sub_rdm_path):
                raise ValueError(f"RDM for subject {subject:02d} not found. Please compute it first.")
            sub_rdm = rdm_instance.load(sub_rdm_path)
            brain_activity.append(sub_rdm)
        brain_activity = np.stack(brain_activity, axis=0)
        logger.info(f"Brain activity loaded successfully!")
        for key in slicing_indices_meg.keys():
            brain_activity_mini = brain_activity[slicing_indices_meg[key]]
            noise_ceiling_instance = noise_ceiling(brain_activity_mini)
            del brain_activity_mini
            logger.info(f"Computing noise ceiling...")
            upper_noise_ceiling, lower_noise_ceiling = noise_ceiling_instance[args.noise_ceiling_type]
            np.save(os.path.join(noise_ceiling_folder, "avg_nc", f"{args.meg_picks}_{key}_upper_noise_ceiling_{args.time_segment}_{args.time_window}.npy"), upper_noise_ceiling)
            np.save(os.path.join(noise_ceiling_folder, "avg_nc", f"{args.meg_picks}_{key}_lower_noise_ceiling_{args.time_segment}_{args.time_window}.npy"), lower_noise_ceiling)
            del upper_noise_ceiling, lower_noise_ceiling
            logger.info(f"Noise ceiling computed successfully!")
    elif args.brain_analysis_type == "raw":
        rois = mne.read_labels_from_annot(parc='aparc_sub', subject='fsaverage', subjects_dir=subjects_dir)
        list_of_regions = [label.name for _, label in enumerate(rois)]
        list_of_regions.remove("unknown-lh")
        list_of_regions.remove("unknown-rh")
        for region_index in range(len(list_of_regions)):
            region = list_of_regions[region_index]
            if args.freq_band is not None:
                region = f"{region}_{args.freq_band}"
            brain_activity = []
            if os.path.exists(os.path.join(noise_ceiling_folder, "raw_nc", f"{region}_scrambled_upper_noise_ceiling.npy")):
                continue
            for subject in range(1, 17):
                logger.info(f"Loading brain activity for subject {subject:02d}...")
                sub_rdm_path = os.path.join(rdms_folder, f"sub-{subject:02d}", "raw_rdm", f"sub-{subject:02d}_meg_{region}_{args.brain_analysis_type}_rdm.npy")
                if not os.path.exists(sub_rdm_path):
                    logger.info(f"RDM for subject {subject:02d} {region} not found. Please compute it first.")
                    continue
                sub_rdm = rdm_instance.load(sub_rdm_path)
                brain_activity.append(sub_rdm)
            brain_activity = np.stack(brain_activity, axis=0)
            logger.info(f"Brain activity loaded successfully!")
            for key in slicing_indices_meg.keys():
                if os.path.exists(os.path.join(noise_ceiling_folder, "raw_nc", f"{region}_{key}_upper_noise_ceiling.npy")):
                    continue
                logger.info(f"Computing noise ceiling for region {region}, stimuli {key}...")
                brain_activity_mini = brain_activity[slicing_indices_meg[key]]
                noise_ceiling_instance = noise_ceiling(brain_activity_mini)
                del brain_activity_mini
                logger.info(f"Computing noise ceiling...")
                upper_noise_ceiling, lower_noise_ceiling = noise_ceiling_instance[args.noise_ceiling_type]
                np.save(os.path.join(noise_ceiling_folder, "raw_nc", f"{region}_{key}_upper_noise_ceiling.npy"), upper_noise_ceiling)
                np.save(os.path.join(noise_ceiling_folder, "raw_nc", f"{region}_{key}_lower_noise_ceiling.npy"), lower_noise_ceiling)
                del upper_noise_ceiling, lower_noise_ceiling, noise_ceiling_instance
                logger.info(f"Noise ceiling computed successfully!")