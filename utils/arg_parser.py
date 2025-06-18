import argparse
from typing import Union

def get_training_config_parser():
    """
    Create an argparse parser for configuring model training.
    """
    parser = argparse.ArgumentParser(description="Model evaluation on datasets.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset",
        type=str,
        default="celebA",
        help="Dataset to use."
    )
    data.add_argument(
        "--analysis_type",
        type=str,
        help="Analysis type for the dataset."
    )
    data.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size."
    )
    data.add_argument(
        "--num_classes",
        type=int,
        default=10000,
        help="Number of classes."
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        choices=[
            "cornet_s", "resnet50", "mobilenet", "vgg16_bn",
            "inception_v3", "FaceNet", "SphereFace",
        ],
        default="resnet50",
        help="Name of the model to run."
    )
    model.add_argument(
        "--n_input_channels",
        type=int,
        choices=[1, 3],
        default=3,
        help="Number of input channels (1 for grayscale, 3 for RGB)."
    )
    model.add_argument(
        "--pretrained",
        dest="pretrained", 
        action="store_true",
        help="If we want to use pretrained weights.",
    )
    model.add_argument(
        "--no-pretrained", 
        dest="pretrained", 
        action="store_false",
        help="Without pretrained weights."
    )
    model.add_argument(
        "--transfer",
        dest="transfer", 
        action="store_true",
        help="If we want to transfer from 3D first layer to 1D.",
    )
    model.add_argument(
        "--no-transfer", 
        dest="transfer", 
        action="store_false",
        help="If we do not want to transfer from 3D first layer to 1D.",
    )
    model.add_argument(
        "--in_weights",
        type=str,
        default=None,
        help="Input weights file for pretrained models.",
    )
    model.add_argument(
        "--out_weights",
        type=str,
        default=None,
        help="Output weights file to save the trained model.",
    )
    model.add_argument(
        '--activ_type', 
        type=str, 
        default= "trained",
        help='Activations type: trained, untrained...'
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of epochs for training."
    )
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="momentum",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="Optimizer choice."
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        choices=[1e-1, 1e-2, 1e-3],
        help="Learning rate for optimizer."
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer."
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay."
    )
    optimization.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size."
    )
    optimization.add_argument(
        "--gamma",
        type=int,
        default=0.1,
        help="Gamma value."
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    loss = parser.add_argument_group("loss")

    loss.add_argument( "--loss_function", choices=["center_loss", "arcface", "cosface", "sphereface","circle_softplus","circle_exp","ContrastiveLoss","TripletLoss"],help="Loss function to use."  )
    triplet_loss_params =parser.add_argument_group("triplet loss param")
    triplet_loss_params.add_argument('--marginTriplet', type=float, default=0.2, help='Margin for Triplet loss')
    circle_loss_params = parser.add_argument_group("CircleLossSoftplus Parameters")
    circle_loss_params.add_argument("--circle_m", type=float, default=0.25, help="Margin parameter for CircleLossSoftplus.")
    circle_loss_params.add_argument("--circle_gamma", type=float, default=256.0, help="Gamma parameter for CircleLossSoftplus.")
    
    angular_loss_params = parser.add_argument_group("Angular Loss Parameters")
    angular_loss_params.add_argument("--s", type=float, default=30.0, help="Scale parameter for angular losses.")
    angular_loss_params.add_argument("--m", type=float, default=0.5, help="Margin parameter for angular losses.")

    center_loss_weight = parser.add_argument_group("center_loss_weight")
    center_loss_weight.add_argument( "--center_loss_weight",type=float, default=1, help="Weight for center loss if used with cross entropy.")
    circle_exp_params=parser.add_argument_group("circle_exp_parameters")
    circle_exp_params.add_argument( "--scale", type=float,default=32,help="Scale parameter for Circle Loss Exp." )
    circle_exp_params.add_argument( "--margin", type=float,default=0.25, help="Margin parameter for Circle Loss Exp.")
    circle_exp_params.add_argument( "--similarity",type=str,default="cos",choices=["cos", "dot"],help="Similarity type for Circle Loss Exp.")


    return parser


def get_similarity_parser():
    """
    Create an argparse parser for configuring similarity analysis.
    """

    parser = argparse.ArgumentParser(description='Compute similarity values for a model')

    activ = parser.add_argument_group("Activations")
    activ.add_argument(
        '--modality',
        type=str,
        choices=[ 
            "brain", "ANN",
            ],
        default="brain",
        help='Modality to use for RDM computation'
    )

    ann = parser.add_argument_group("ANN")
    ann.add_argument(
        "--model_name",
        type=str,
        choices=[ 
                 "cornet_s", "resnet50",
                 "mobilenet",  "vgg16_bn", 
                 "inception_v3", "FaceNet", "SphereFace",
                 ],
        default="resnet50",
        help="name of the model to run."
    )
    ann.add_argument(
        '--ann_analysis_type', 
        type=str, 
        choices= [
            "with-meg_pretrained", "without-meg_pretrained", "None",
            "random_faces", "meg_stimuli_faces"
        ],
        default = "None",
        help='For which analysis type to compute RDMs'
    )
    ann.add_argument(
        '--activation_type', 
        type=str, 
        choices= [
            "trained", "untrained", "imagenet", "VGGFace"
        ],
        default= "trained",
        help='Activations type: trained, untrained'
    )
    ann.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    ann.add_argument(
        '--index',
        type=int,
        default=0,
        help='Index to compute RDMs'
    )

    brain = parser.add_argument_group("Brain Data")
    brain.add_argument(
        '--subject', 
        type=int, 
        default= 1,
        help='Subject id'
    )
    brain.add_argument(
        '--brain_analysis_type',
        type=str, 
        choices= [ 
            "avg", "raw"
        ],
        default= "avg",
        help='Brain analysis type: avg, raw.'
    )
    brain.add_argument(
        '--time_window', 
        type=int, 
        default= 50,
        help='Time window for across time RDM computing...'
    )
    brain.add_argument(
        '--meg_picks',
        type=str, 
        default="grad",
        help='Type of MEG sensors to use'
    )
    brain.add_argument(
        '--stimuli_file_name',
        type=str,
        default="fam",
        help='File name for stimuli images'
    )
    brain.add_argument(
        '--time_segment',
        type=int,
        default=550,
        help='Time segment to compute RDM of brain activity'
    )
    brain.add_argument(
        '--sliding_window',
        type=int,
        default=50,
        help='Sliding window to compute RDM of brain activity'
    )
    brain.add_argument(
        '--region_index',
        type=int,
        default=0,
        help='Region index to compute RDM of brain activity'
    )
    brain.add_argument(
        '--freq_band',
        type=Union[str, bool],
        default=None,
        help='Frequency band for source reconstruction'
    )
    brain.add_argument(
        "--subject_rdms_folder",
        type=str,
        default=None,
        help="Path to the RDM folder."
    )
    parser.add_argument(
        "--raw_mode",
        choices=["each", "avg"],
        default="each",
        help="In raw mode: 'each' for per‐subject scores, 'avg' to average across subjects first"
    )

    stats = parser.add_argument_group("Stats")
    stats.add_argument(
        '--similarity_measure', 
        type=str, 
        default="spearman",
        help='Similarity measure to use'
    )
    stats.add_argument(
        '--noise_ceiling_type',
        type=str,
        choices= [
            "bootstrap", "loo"
        ],
        default= "loo",
        help='Noise ceiling type: bootstrap, loo.'
    )

    overwrite = parser.add_argument_group("Overwrite")
    overwrite.add_argument(
        "--overwrite",
        dest="overwrite", 
        action="store_true",
        help="If we want to overwrite existing files.",
    )

    overwrite.add_argument(
        "--no-overwrite", 
        dest="overwrite", 
        action="store_false",
        help="Without overwrite."
    )

    return parser


def source_rescontruction_parser():
    """
    Create an argparse parser for configuring source reconstrcution.
    """
    parser = argparse.ArgumentParser(description="Source Reconstruction.")

    source = parser.add_argument_group("Source Data")
    source.add_argument(
        "--subject", 
        type=int, 
        default=1, 
        help="Subject id."
    )
    source.add_argument(
        "--method", 
        type=str, 
        default="MNE", 
        help="Source Estimation method."
    )
    source.add_argument(
        "--overwrite",
        dest="overwrite", 
        action="store_true",
        help="If we want to overwrite existing files.",
    )
    source.add_argument(
        "--no-overwrite", 
        dest="overwrite", 
        action="store_false",
        help="Without overwrite."
    )
    source.add_argument(
        '--stimuli_file_name',
        type=str, 
        default="Fam",
        help='File name for stimuli images'
    )
    source.add_argument(
        '--meg_picks',
        type=str, 
        default="grad",
        help='Type of MEG sensors to use'
    )

    source.add_argument(
        '--freq_bands',
        type=dict,
        default={
            "theta": (4, 8), 
            "alpha": (8, 12), 
            "beta": (12, 30), 
            "gamma": (30, 55), 
            "high-gamma": (55, 90)
        },
        help='Frequency bands for hilbert transform'
    )


    return parser