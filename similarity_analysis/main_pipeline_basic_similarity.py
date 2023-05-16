import time
import numpy as np
from models.inception import inception_v3
from models.cornet_s import cornet_s
from models.mobilenet import mobilenet_v2
from models.resnet import resnet50
from models.vgg import vgg16_bn
from models.FaceNet import FaceNet
from models.SphereFace import SphereFace
from utils.config import get_similarity_parser
from utils.config_sim_analysis import networks, meg_sensors
from utils.general import load_meg_rdm
from src.extract_activations import extract_activations
from src.compute_models_rdms import extract_network_rdms
from src.similarity import average_similarity_score

if __name__ == '__main__':
    start = time.time()
    parser = get_similarity_parser()
    args = parser.parse_args()
    model_cls = { "cornet_s": cornet_s, "resnet50": resnet50, "mobilenet": mobilenet_v2,  "vgg16_bn": vgg16_bn, 
                 "inception_v3": inception_v3, "FaceNet": FaceNet, "SphereFace": SphereFace}[args.model_name]
    model = model_cls(False, 1000, 1)
    list_layers = networks[args.model_name]
    
    activs = extract_activations(args.cons, args.stimuli_file_name, args.model_name, model, list_layers, args.weights, args.method, args.save, "trained")
    activs = extract_activations(args.cons, args.stimuli_file_name, args.model_name, model, list_layers, "None", args.method, args.save, "untrained")

    meg_rdm = load_meg_rdm(args.stimuli_file_name, args.power)
    meg_rdm = np.mean(meg_rdm, axis=0)

    layers_rdms, network_rdm = extract_network_rdms(args.cons, args.stimuli_file_name, args.model_name, args.save, "trained")
    avg_sim_scores, avg_high_sim_scores, avg_model_sim_scores, avg_model_high_sim_scores = average_similarity_score(meg_rdm, meg_sensors, 
                layers_rdms, network_rdm, list_layers, args.save, model_name = args.model_name, stimuli_file= args.stimuli_file_name, method = ["pearson"], activ_type = args.activ_type)

    layers_rdms, network_rdm = extract_network_rdms(args.cons, args.stimuli_file_name, args.model_name, args.save, "untrained")
    avg_sim_scores, avg_high_sim_scores, avg_model_sim_scores, avg_model_high_sim_scores = average_similarity_score(meg_rdm, meg_sensors, 
                layers_rdms, network_rdm, list_layers, args.save, model_name = args.model_name, stimuli_file= args.stimuli_file_name, method = ["pearson"], activ_type = "untrained")

    time_sim = time.time() - start
    print(f'Computations ended in {time_sim // 3600} h {(time_sim % 3600) // 60} m { time_sim % 60} s')
