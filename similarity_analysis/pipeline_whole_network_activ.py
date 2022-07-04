from config_sim_analysis import networks, meg_rdm, meg_sensors, vgg_layers, inception_layers, mobilenet_layers, SphereFace_layers
from extract_activations import get_main_network_activations, get_whole_network_activations
from networks_rdm import get_network_rdm
from similarity import whole_network_similarity_scores

if __name__ == '__main__':
    """ This function for generating results for the whole network results (activations of all layers concatinated) """
    for network, layers in networks.items():
        print("\nNetwork %s started\n"%network)
        main=get_main_network_activations('%s_FamUnfam_activations'%network, layers)
        whole=get_whole_network_activations('%s_FamUnfam_activations'%network)
        rdm=get_network_rdm('%s_FamUnfam'%network)
        sim=whole_network_similarity_scores('%s_FamUnfam'%network, meg_rdm, meg_sensors)
        print('\n%s done\n'%network)
