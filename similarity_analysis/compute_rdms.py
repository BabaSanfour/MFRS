import os
import json
import pickle
import time
from datetime import datetime 
import numpy as np
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rdm import RDM
from utils.arg_parser import get_similarity_parser
from utils.config import activations_folder, rdms_folder, meg_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = get_similarity_parser()
    args = parser.parse_args()
    rdm = RDM(428)
    if args.modality == "ANN":
        with open(os.path.join(activations_folder, f"{args.model_name}_{args.ann_analysis_type}_activations_{args.activation_type}.pkl"), 'rb') as pickle_file:
            activations = pickle.load(pickle_file)

        with open(os.path.join(meg_dir, "mega_events_disregarded.json"), 'rb') as json_file:
            events = json.load(json_file)
        events_to_remove = np.array(events['all']).astype(int)
        mask = np.isin(np.arange(450), events_to_remove, invert=True)
        for layer_name, layer_activations in activations.items():
            updated_layer_activations = layer_activations[mask]
            activations[layer_name] = updated_layer_activations

        logger.info(f"Calculating RDMs for {args.model_name} model, {args.ann_analysis_type} analysis, {args.activation_type} activations...")
        rdm.save(rdm.model_rdms_parallel(activations), os.path.join(rdms_folder, f"{args.model_name}_{args.ann_analysis_type}_rdm_{args.activation_type}.npy"))
        logger.info("RDMs computed and saved successfully!")
    elif args.modality == "brain":
        subject = f"sub-{args.subject:02d}"
        with open(os.path.join(meg_dir, subject, f"{subject}-{args.meg_picks}-ROI-trans-{args.brain_analysis_type}-time-courses.pkl"), 'rb') as pickle_file:
            brain_activity = pickle.load(pickle_file)
        if args.brain_analysis_type == "avg":
            logger.info(f"Calculating temporal brain RDMs for subject {args.subject:02d}...")
            rdm.save(rdm.temp_brain_rdms_parallel(brain_activity, args.time_segment, args.sliding_window), os.path.join(rdms_folder, f"{subject}_{args.meg_picks}_{args.brain_analysis_type}_rdm_{args.time_segment}_{args.sliding_window}.npy"))
        elif args.brain_analysis_type == "raw":
            if not os.path.isdir(os.path.join(rdms_folder, subject)):
                os.mkdir(os.path.join(rdms_folder, subject))
            list_of_regions = list(brain_activity.keys())
            list_of_regions.remove("unknown-lh")
            list_of_regions.remove("unknown-rh")
            region = list_of_regions[args.region_index]
            region_activity = brain_activity[region]
            del brain_activity
            if os.path.exists(os.path.join(rdms_folder, subject, f"{subject}_{region}_{args.brain_analysis_type}_rdm.npy")):
                logger.info(f"RDM movie for {region} already exists. Skipping...")
            else:
                logger.info(f"Calculating brain RDM movie for subject {args.subject:02d} for {region}...")
                start_time = time.time()
                rdm.save(rdm.brain_rdm_movie_parallel({region: region_activity}), os.path.join(rdms_folder, subject, f"{subject}_{region}_{args.brain_analysis_type}_rdm.npy"))
                end_time = time.time()
                time_taken = datetime.utcfromtimestamp(end_time - start_time).strftime("%H:%M:%S")
                logger.info(f"RDM movie for {region} computed and saved successfully!")
                logger.info(f"Time taken: {time_taken}")
        else :
            raise ValueError("Brain analysis type not recognized. Please choose between 'avg' and 'raw'")
        logger.info("RDMs computed and saved successfully!")
    else:
        raise ValueError("Modality not recognized. Please choose between 'brain' and 'ANN'")


    