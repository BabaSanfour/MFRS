import os
import mne
import numpy as np
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rdm import RDM
from src.rsa import RSA
from utils.arg_parser import get_similarity_parser
from utils.config import rdms_folder, similarity_folder, subjects_dir, rois_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_avg(args, rsa_instance, rdm_instance):
    """Compute similarity using the average‐across‐subjects RDM."""
    # build output filename
    picks = args.meg_picks
    if args.freq_band:
        picks = f"{picks}_{args.freq_band}"
    out_fname = os.path.join(
        similarity_folder,
        f"avg_{args.model_name}_{args.ann_analysis_type}_{picks}_rdm_"
        f"{args.activation_type}_{args.stimuli_file_name}_"
        f"{args.time_segment}_{args.sliding_window}.npy"
    )
    if os.path.exists(out_fname):
        logger.info(f"[avg] Output already exists at {out_fname}, skipping.")
        return

    # load or compute brain RDM
    rdm_path = os.path.join(rdms_folder, f"{picks}_avg_rdm_{args.time_segment}_{args.sliding_window}.npy")
    if os.path.exists(rdm_path):
        brain_rdm = rdm_instance.load(rdm_path)
        logger.info(f"[avg] Loaded existing avg RDM for {picks}.")
    else:
        logger.info(f"[avg] Computing avg RDM across all subjects for {picks}...")
        all_subs = []
        for i in range(1, 17):
            sub_path = os.path.join(
                rdms_folder,
                f"sub-{i:02d}_{picks}_avg_rdm_{args.time_segment}_{args.sliding_window}.npy"
            )
            if not os.path.exists(sub_path):
                raise FileNotFoundError(f"Missing RDM for subject {i:02d} at {sub_path}")
            all_subs.append(rdm_instance.load(sub_path))
        brain_rdm = np.mean(np.stack(all_subs, axis=0), axis=0)
        rdm_instance.save(brain_rdm, rdm_path)
        logger.info(f"[avg] Saved newly computed avg RDM at {rdm_path}.")

    # slice and score
    brain_rdm = brain_rdm[slicing_indices_meg[args.stimuli_file_name]]
    model_path = os.path.join(
        rdms_folder,
        f"{args.model_name}_{args.ann_analysis_type}_rdm_{args.activation_type}.npy"
    )
    model_rdm = rdm_instance.load(model_path)[slicing_indices_model[args.stimuli_file_name]]
    sim = rsa_instance.score(brain_rdm, model_rdm)
    rsa_instance.save(sim, out_fname)
    logger.info(f"[avg] Similarity score saved to {out_fname}.")


def process_raw(args, rsa_instance, rdm_instance):
    """
    Compute similarity per‐region:
      - raw_mode=='each': loop subjects 1–16, one RSA each
      - raw_mode=='avg' : average all 16 RDMs, then one RSA per region
    """
    regions = [r for r in rois_names if not r.startswith("unknown-")]
    model_path = os.path.join(
        rdms_folder, "networks_rdms",
        f"{args.model_name}_{args.ann_analysis_type}_rdm_{args.activation_type}.npy"
    )
    model_rdm = rdm_instance.load(model_path)

    out_base = os.path.join(similarity_folder, "raw_subject_sim_scores")
    os.makedirs(out_base, exist_ok=True)

    # If avg across subjects, load all subs once per region
    if args.raw_mode == "avg":
        for region in regions:
            reg_key = f"{region}_{args.freq_band}" if args.freq_band else region
            # gather all subjects' RDMs
            subs = []
            for subj in range(1, 17):
                path = os.path.join(
                    args.subject_rdms_folder,
                    f"sub-{subj:02d}_{reg_key}_raw_rdm.npy"
                )
                if os.path.exists(path):
                    subs.append(rdm_instance.load(path))
            if not subs:
                logger.info(f"[raw|avg] No RDMs found for region {reg_key}, skipping.")
                continue

            brain_rdm = np.mean(np.stack(subs, axis=0), axis=0)
            for stim in ("fam", "unfamiliar", "scrambled"):
                out_dir = os.path.join(out_base, "avg", reg_key, stim)
                os.makedirs(out_dir, exist_ok=True)
                out_fname = os.path.join(
                    out_dir,
                    f"raw_avg_{args.model_name}_{args.ann_analysis_type}_"
                    f"{args.activation_type}_{stim}.npy"
                )
                if os.path.exists(out_fname):
                    logger.info(f"[raw|avg] {out_fname} exists, skipping.")
                    continue

                br = brain_rdm[slicing_indices_meg[stim]]
                mr = model_rdm[slicing_indices_model[stim]]
                sim = rsa_instance.score(br, mr)
                rsa_instance.save(sim, out_fname)
                logger.info(f"[raw|avg] Saved average‐subject similarity for {reg_key}/{stim}")

    # Per‐subject mode
    else:  # args.raw_mode == "each"
        for subj in range(1, 17):
            subj_base = os.path.join(out_base, f"sub-{subj:02d}")
            os.makedirs(subj_base, exist_ok=True)

            for region in regions:
                reg_key = f"{region}_{args.freq_band}" if args.freq_band else region
                brain_path = os.path.join(
                    args.subject_rdms_folder,
                    f"sub-{subj:02d}_{reg_key}_raw_rdm.npy"
                )
                if not os.path.exists(brain_path):
                    logger.info(f"[raw|each] No RDM for sub-{subj:02d}, region {reg_key}, skipping.")
                    continue

                brain_rdm = rdm_instance.load(brain_path)
                for stim in ("fam", "unfamiliar", "scrambled"):
                    out_dir = os.path.join(
                        subj_base,
                        f"sub-{subj:02d}_{args.model_name}_"
                        f"{args.ann_analysis_type}_{args.activation_type}_{stim}"
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    out_fname = os.path.join(
                        out_dir,
                        f"raw_{args.model_name}_{args.ann_analysis_type}_"
                        f"{region}_{args.activation_type}_{stim}.npy"
                    )
                    if os.path.exists(out_fname):
                        logger.info(f"[raw|each] {out_fname} exists, skipping.")
                        continue

                    br = brain_rdm[slicing_indices_meg[stim]]
                    mr = model_rdm[slicing_indices_model[stim]]
                    sim = rsa_instance.score(br, mr)
                    rsa_instance.save(sim, out_fname)
                    logger.info(
                        f"[raw|each] Saved similarity for sub-{subj:02d}, "
                        f"{region}/{stim}"
                    )


if __name__ == "__main__":
    parser = get_similarity_parser()
    args = parser.parse_args()
    rsa = RSA(428)
    rdm = RDM(428)

    # define your slicing indices once
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

    logger.info("Starting similarity analysis...")
    if args.brain_analysis_type == "avg":
        process_avg(args, rsa, rdm)
    elif args.brain_analysis_type == "raw":
        process_raw(args, rsa, rdm)
    else:
        raise ValueError("Modality not recognized. Please choose 'avg' or 'raw'.")