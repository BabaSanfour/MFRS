import argparse

def get_training_config_parser():
    """
    Create an argparse parser for configuring model training.

    Returns:
        argparse.ArgumentParser: A parser for training configuration.
    """
    parser = argparse.ArgumentParser(description="Model evaluation on datasets.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset",
        type=str,
        default="celebA",
        help="Dataset to use (default: %(default)s)."
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
        help="Batch size (default: %(default)s)."
    )
    data.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of classes (default: %(default)s).",
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
        help="Name of the model to run (default: %(default)s).",
    )
    model.add_argument(
        "--n_input_channels",
        type=int,
        choices=[1, 3],
        default=1,
        help="Number of input channels (1 for grayscale, 3 for RGB, default: %(default)s).",
    )

    model.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Use pretrained weights (default: %(default)s).",
    )
    model.add_argument(
        "--transfer",
        type=bool,
        default=True,
        help="Transfer learning (default: %(default)s).",
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

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs for training (default: %(default)s).",
    )
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="momentum",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="Optimizer choice (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        choices=[1e-1, 1e-2, 1e-3],
        help="Learning rate for Adam optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: %(default)s).",
    )
    optimization.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size (default: %(default)s).",
    )
    optimization.add_argument(
        "--gamma",
        type=int,
        default=0.1,
        help="Gamma value (default: %(default)s).",
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatability (default: %(default)s).",
    )

    return parser


def get_similarity_parser():
    parser = argparse.ArgumentParser(description='Compute similarity values for a model')
    activ = parser.add_argument_group("Activations")
    activ.add_argument(
        '--cons',
        type=int,
        default=300,
        help='Number of images to use as stimuli'
    )
    activ.add_argument(
        '--stimuli_file_name',
        type=str, 
        default="FamUnfam",
        help='File name for stimuli images'
    )
    activ.add_argument(
        '--band',
        type=str, 
        default=None,
        help='Name of power band'
    )

    activ.add_argument(
        "--model_name",
        type=str,
        choices=[ 
                 "cornet_s", "resnet50",
                 "mobilenet",  "vgg16_bn", 
                 "inception_v3", "FaceNet", "SphereFace",
                 ],
        default="resnet50",
        help="name of the model to run (default: %(default)s).",
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        help='Path to the model weights file'
    )
    parser.add_argument(
        '--method', 
        type=str, 
        choices = ["all", "index", "list", "type"],
        default = "list",
        help='Method to use for activation extraction'
    )
    parser.add_argument(
        '--save',
        default = True, 
        action='store_true', 
        help='Save the extracted activations to disk'
    )
    parser.add_argument(
        '--activ_type', 
        type=str, 
        default= "trained",
        help='Activations type: trained, untrained...'
    )
    parser.add_argument(
        '--analysis_type', 
        type=str, 
        default= "basic",
        choices = ["basic", "across_time"],
        help='Type of MEG RDMs...'
    )
    parser.add_argument(
        '--time_window', 
        type=int, 
        default= 5,
        help='Time window for across time RDM computing...'
    )

    return parser

def source_rescontruction_parser():
    parser = argparse.ArgumentParser(description="Source Reconstruction.")

    source = parser.add_argument_group("SR")
    source.add_argument(
        "--subject", 
        type=int, 
        default=1, 
        help="Subject id (default: %(default)s)."
    )

    source.add_argument(
        "--method", 
        type=str, 
        default="MNE", 
        help="Source Estimation method (default: %(default)s)."
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

    return parser


