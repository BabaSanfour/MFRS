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

    if args.brain_analysis_type == "avg":
        logger.info(f"Loading brain activity...")
        brain_activity = []
        for i in range(1,17):
            sub_rdm_path = os.path.join(rdms_folder, f"sub-{i:02d}_avg_rdm.npy")
            sub_rdm = rdm_instance.load(sub_rdm_path)
            if i ==14:
                brain_activity.append(sub_rdm)
            else:
                brain_activity.append(np.transpose(sub_rdm, (1, 0, 2, 3)))
        brain_activity = np.stack(brain_activity, axis=0)
        brain_activity = np.mean(brain_activity, axis=0)
        brain_activity = brain_activity[:, :, 0:125, 0:125]
        logger.info(f"Brain activity loaded successfully!")
        logger.info(f"loading {args.model_name} model RDM...")
        model_rdm_path = os.path.join(rdms_folder, f"{args.model_name}_{args.ann_analysis_type}_rdm_{args.activation_type}.npy")
        model_rdm = rdm_instance.load(model_rdm_path)
        model_rdm = model_rdm[:, 0:125, 0:125]
        logger.info(f"{args.model_name} model RDM loaded successfully!")
        logger.info(f"Computing similarity score...")
        sim = rsa_instance.boostrap(brain_activity, model_rdm)
        rsa_instance.save(sim, os.path.join(similarity_folder, f"avg_{args.model_name}_{args.ann_analysis_type}_rdm_{args.activation_type}.npy"))
        logger.info(f"Similarity score computed and saved successfully!")
    elif args.brain_analysis_type == "raw":
        if not os.path.isdir(os.path.join(rdms_folder, subject)):
            os.mkdir(os.path.join(rdms_folder, subject))
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


    