import os
import json
import pickle
import numpy as np
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rdm import RDM
from src.rsa import RSA
from utils.arg_parser import get_similarity_parser
from utils.config import activations_folder, rdms_folder, meg_dir, similarity_folder

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
        
        logger.info(f"Loading brain activity...")
        for time_segment, nbr_rdms in zip(["330", "440", "550", "660", "770"], [12, 9, 7, 5, 3]):
            brain_activity = np.zeros((450, nbr_rdms, 428, 428))
            rdm_path = os.path.join(rdms_folder, f"{args.meg_picks}_avg_rdm_{time_segment}_{args.sliding_window}.npy")
            if os.path.exists(rdm_path):
                logger.info(f"RDM for {args.meg_picks} already exists. Loading...")
                brain_activity = rdm_instance.load(rdm_path)
            else:
                logger.info(f"RDM for {args.meg_picks}, {time_segment}, {args.sliding_window} not found. Computing...")
                for i in range(1,17):
                    sub_rdm_path = os.path.join(rdms_folder, f"sub-{i:02d}_{args.meg_picks}_avg_rdm_{time_segment}_{args.sliding_window}.npy")
                    if not os.path.exists(sub_rdm_path):
                        raise ValueError(f"RDM for subject {i:02d} not found. Please compute it first.")
                    sub_rdm = rdm_instance.load(sub_rdm_path)
                    brain_activity  += sub_rdm
                    print("subject ", i, " loaded")
                
                brain_activity /= 16
                rdm_instance.save(brain_activity, rdm_path)
            logger.info(f"Brain activity loaded successfully!")
            ann_analysis_types = ["None", "with-meg_pretrained", ]#"without-meg_pretrained", "None", "None", "meg_stimuli_faces", "random_faces"]
            activations = ["untrained", "trained", ]#"trained", "imagenet", "VGGFace", "imagenet", "imagenet"]
            for model_name in  ["FaceNet"]:#, "SphereFace", "resnet50", "vgg16_bn", "cornet_s", "mobilenet", "inception_v3"]:
                for ann_analysis_type, activation_type in zip(ann_analysis_types, activations):
                    if os.path.exists(os.path.join(rdms_folder, f"{model_name}_{ann_analysis_type}_rdm_{activation_type}.npy")):
                        logger.info(f"{model_name} {ann_analysis_type} {activation_type} model RDM exists. Loading...")
                        model_rdm = rdm_instance.load(os.path.join(rdms_folder, f"{model_name}_{ann_analysis_type}_rdm_{activation_type}.npy"))
                    else:
                        raise ValueError(f"{model_name} model RDM not found. Please compute it first.")
                    logger.info(f"{model_name} model RDM loaded successfully!")
                    for stimuli_file_name in ["fam", "unfamiliar", "scrambled"]:
                        # if os.path.exists(os.path.join(similarity_folder, f"avg_{model_name}_{ann_analysis_type}_{args.meg_picks}_rdm_{activation_type}_{stimuli_file_name}_{time_segment}_{args.sliding_window}.npy")):
                        #     logger.info(f"Similarity score for {model_name} and {stimuli_file_name} already exists. Skipping...")
                        #     continue
                        selected_brain_activity = brain_activity[slicing_indices_meg[stimuli_file_name]]
                        selected_model_rdm = model_rdm[slicing_indices_model[stimuli_file_name]]
                        logger.info(f"Computing similarity score...")
                        sim = rsa_instance.score(selected_brain_activity, selected_model_rdm)
                        rsa_instance.save(sim, os.path.join(similarity_folder, f"avg_{model_name}_{ann_analysis_type}_{args.meg_picks}_rdm_{activation_type}_{stimuli_file_name}_{time_segment}_{args.sliding_window}.npy"))
                        logger.info(f"Similarity score computed and saved successfully!")
    elif args.brain_analysis_type == "raw":
        if not os.path.isdir(os.path.join(rdms_folder, args.subject)):
            os.mkdir(os.path.join(rdms_folder, args.subject))
        logger.info(f"Loading brain activity...")
        for region, region_activity in brain_activity.items():
            logger.info(f"Calculating brain RDM movie for subject {args.subject:02d} for {region}...")
            if os.path.exists(os.path.join(rdms_folder, subject, f"{subject}_{region}_{args.brain_analysis_type}_rdm.npy")):
                logger.info(f"RDM movie for {region} already exists. Skipping...")
                continue
            rdm.save(rdm.brain_rdm_movie_parallel({region: region_activity}), os.path.join(rdms_folder, subject, f"{subject}_{region}_{args.brain_analysis_type}_rdm.npy"))
            logger.info(f"RDM movie for {region} computed and saved successfully!")

        else :
            raise ValueError("Brain analysis type not recognized. Please choose between 'avg' and 'raw'")
        logger.info("RDMs computed and saved successfully!")
    else:
        raise ValueError("Modality not recognized. Please choose between 'brain' and 'ANN'")


    