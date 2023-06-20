# configuration file

import argparse

def get_config_parser():
    parser = argparse.ArgumentParser(description="models evalution on Face datasets.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset", type=str, default="celebA", help="Dataset to use (default: %(default)s)."
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--batch_size", 
        type=int, 
        default=32, #fixed after Random Search
        choices=[8, 16, 32, 64, 128, 256, 512],
        help="batch size (default: %(default)s)."
    )
    data.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        choices=[1000, 500, 300],
        help="Use VGGFace pretrained weights (default: %(default)s).",
    )
    data.add_argument(
        "--num_pictures",
        type=int,
        default=30,
        help="Num pictures per class for CelebA (default: %(default)s).",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        choices=["alexnet", "resnet18" "resnet34", "deepID_1", 
                "LightCNN", "cornet_z", "cornet_s", "resnet50",
                 "mobilenet",  "vgg16_bn", 
                 "vgg19_bn", "inception_v3", "FaceNet", "SphereFace",
                 "resnet101", "resnet152", "vgg16", "vgg19"],
        default="resnet50",
        help="name of the model to run (default: %(default)s).",
    )
    model.add_argument(
        "--n_input_channels",
        type=int,
        choices=[1, 3],
        default=1,
        help="RGB or GreyScale pictures as input (default: %(default)s).",
    )

    model.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Use VGGFace pretrained weights (default: %(default)s).",
    )
    model.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights to use in case of pretrained=true (default: %(default)s).",
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="number of epochs for training (default: %(default)s).",
    )
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="momentum",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="choice of optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-2, #fixed after random search
        choices=[1e-1, 1e-2, 1e-3],
        help="learning rate for Adam optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay (default: %(default)s).",
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
        help="Step size (default: %(default)s).",
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for repeatability (default: %(default)s).",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to store tensors on (default: %(default)s).",
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
        '--type_meg_rdm', 
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



