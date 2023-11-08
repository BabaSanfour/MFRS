import argparse

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
        default=32,
        help="Batch size."
    )
    data.add_argument(
        "--num_classes",
        type=int,
        default=1000,
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
        default=1,
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
        default=50,
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
        help="Random seed for repeatability."
    )

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
            "trained", "untrained", "imagenet"
        ],
        default= "trained",
        help='Activations type: trained, untrained'
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

    return parser