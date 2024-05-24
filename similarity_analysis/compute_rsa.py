import os
import mne
import numpy as np
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rdm import RDM
from src.rsa import RSA
from utils.arg_parser import get_similarity_parser
from utils.config import rdms_folder, similarity_folder, subjects_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = get_similarity_parser()
    args = parser.parse_args()
    rsa_instance = RSA(428)
    rdm_instance = RDM(428)
    slicing_indices_meg = {
        "fam": (slice(None), slice(None), slice(130), slice(130)),
        "unfamiliar": (slice(None), slice(None), slice(130, 280), slice(130, 280)),
        "scrambled": (slice(None), slice(None), slice(280, None), slice(280, None))
    }

    slicing_indices_model = {
        "fam": (slice(None), slice(130), slice(130)),
        "unfamiliar": (slice(None), slice(130, 280), slice(130, 280)),
        "scrambled": (slice(None), slice(280, None), slice(280, None))
    }

    if args.brain_analysis_type == "avg":
        if args.freq_band is not None:
            args.meg_picks = f"{args.meg_picks}_{args.freq_band}"
        if os.path.exists(os.path.join(similarity_folder, f"avg_{args.model_name}_{args.ann_analysis_type}_{args.meg_picks}_rdm_{args.activation_type}_{args.stimuli_file_name}_{args.time_segment}_{args.sliding_window}.npy")):
            logger.info(f"Similarity score for {args.model_name}, {args.stimuli_file_name} {args.ann_analysis_type}, {args.meg_picks}, {args.activation_type}, {args.time_segment}, {args.sliding_window} already exists. Skipping...")
        else:
            logger.info(f"Loading brain rdm...")
            brain_rdm = []
            rdm_path = os.path.join(rdms_folder, f"{args.meg_picks}_avg_rdm_{args.time_segment}_{args.sliding_window}.npy")
            if os.path.exists(rdm_path):
                logger.info(f"RDM for {args.meg_picks} already exists. Loading...")
                brain_rdm = rdm_instance.load(rdm_path)
            else:
                logger.info(f"RDM for {args.meg_picks}, {args.time_segment}, {args.sliding_window} not found. Computing...")
                for i in range(1,17):
                    sub_rdm_path = os.path.join(rdms_folder, f"sub-{i:02d}_{args.meg_picks}_avg_rdm_{args.time_segment}_{args.sliding_window}.npy")
                    if not os.path.exists(sub_rdm_path):
                        raise ValueError(f"RDM for subject {i:02d} not found. Please compute it first.")
                    sub_rdm = rdm_instance.load(sub_rdm_path)
                    brain_rdm.append(sub_rdm)
                brain_rdm = np.stack(brain_rdm, axis=0)
                brain_rdm = np.mean(brain_rdm, axis=0)
                rdm_instance.save(brain_rdm, rdm_path)
            brain_rdm = brain_rdm[slicing_indices_meg[args.stimuli_file_name]]
            logger.info(f"brain rdm loaded successfully!")
            logger.info(f"loading {args.model_name} model RDM...")
            model_rdm_path = os.path.join(rdms_folder, f"{args.model_name}_{args.ann_analysis_type}_rdm_{args.activation_type}.npy")
            model_rdm = rdm_instance.load(model_rdm_path)
            model_rdm = model_rdm[slicing_indices_model[args.stimuli_file_name]]
            logger.info(f"{args.model_name} model RDM loaded successfully!")
            logger.info(f"Computing similarity score...")
            sim = rsa_instance.score(brain_rdm, model_rdm)
            rsa_instance.save(sim, os.path.join(similarity_folder, f"avg_{args.model_name}_{args.ann_analysis_type}_{args.meg_picks}_rdm_{args.activation_type}_{args.stimuli_file_name}_{args.time_segment}_{args.sliding_window}.npy"))
            logger.info(f"Similarity score computed and saved successfully!")
    elif args.brain_analysis_type == "raw":
        rois = mne.read_labels_from_annot(parc='aparc_sub', subject='fsaverage', subjects_dir=subjects_dir)
        list_of_regions = [label.name for _, label in enumerate(rois)]
        list_of_regions.remove("unknown-lh")
        list_of_regions.remove("unknown-rh")
        next_region = False
        for region_index, region in enumerate(list_of_regions):
            region = list_of_regions[region_index]
            logger.info(f"Calculating brain Similarity Scores movie for region {region}...")
            brain_rdm = np.zeros((1, 1101, 428, 428), dtype=np.float32)
            if args.freq_band is not None:
                region = f"{region}_{args.freq_band}"

            rdm_path = os.path.join(rdms_folder, "raw_rdms_sbjct_avg", f"{region}_raw_rdm.npy")
            if os.path.exists(rdm_path):
                logger.info(f"RDM for region {region} already exists. Loading...")
                brain_rdm = rdm_instance.load(rdm_path)
                logger.info(f"brain rdm loaded successfully!")
            else:
                logger.info(f"RDM for region {region} not found. Computing...")
                for i in range(1,17):
                    sub_rdm_path = os.path.join(rdms_folder, f"sub-{i:02d}", "raw_rdm", f"sub-{i:02d}_{args.meg_picks}_{region}_raw_rdm.npy")
                    if not os.path.exists(sub_rdm_path):
                        raise ValueError(f"RDM for subject {i:02d}, region {region} not found. Please compute it first.")
                    try:
                        brain_rdm += rdm_instance.load(sub_rdm_path)
                        logger.info(f"RDM for subject {i:02d} loaded successfully!")
                    except:
                        logger.info(f"RDM for subject {i:02d} not found. Moving to next region...")
                        next_region = True
                        break
                    brain_rdm += rdm_instance.load(sub_rdm_path)
                brain_rdm = brain_rdm / 16
                rdm_instance.save(brain_rdm, rdm_path)
                logger.info(f"brain rdm computed and saved successfully!")
            if next_region:
                next_region = False
                logger.info(f"Skipping region {region}...")
                continue
            logger.info(f"loading {args.model_name} model RDM...")
            model_rdm_path = os.path.join(rdms_folder, "networks_rdms", f"{args.model_name}_{args.ann_analysis_type}_rdm_{args.activation_type}.npy")
            model_rdm = rdm_instance.load(model_rdm_path)
            logger.info(f"{args.model_name} model RDM loaded successfully!")
            for stimuli_file_name in ["fam", "unfamiliar", "scrambled"]:
                if os.path.exists(os.path.join(similarity_folder, "raw_sim_scores", f"{args.model_name}_{args.ann_analysis_type}_{args.activation_type}_{stimuli_file_name}", f"raw_{args.model_name}_{args.ann_analysis_type}_{region}_rdm_{args.activation_type}_{stimuli_file_name}.npy")):
                    logger.info(f"Similarity score for {args.model_name}, {args.ann_analysis_type}, {region}, {args.activation_type}, {stimuli_file_name} already exists. Skipping...")
                    continue
                brain_rdm_mini = brain_rdm[slicing_indices_meg[stimuli_file_name]]
                model_rdm_mini = model_rdm[slicing_indices_model[stimuli_file_name]]
                logger.info(f"Computing similarity score...")
                sim = rsa_instance.score(brain_rdm_mini, model_rdm_mini)
                if not os.path.isdir(os.path.join(similarity_folder, "raw_sim_scores", f"{args.model_name}_{args.ann_analysis_type}_{args.activation_type}_{stimuli_file_name}")):
                    os.makedirs(os.path.join(similarity_folder, "raw_sim_scores", f"{args.model_name}_{args.ann_analysis_type}_{args.activation_type}_{stimuli_file_name}"))
                rsa_instance.save(sim, os.path.join(similarity_folder, "raw_sim_scores", f"{args.model_name}_{args.ann_analysis_type}_{args.activation_type}_{stimuli_file_name}", f"raw_{args.model_name}_{args.ann_analysis_type}_{region}_rdm_{args.activation_type}_{stimuli_file_name}.npy"))
                logger.info(f"Similarity score computed and saved successfully!")

    else:
        raise ValueError("Modality not recognized. Please choose between 'avg' and 'raw'")


    