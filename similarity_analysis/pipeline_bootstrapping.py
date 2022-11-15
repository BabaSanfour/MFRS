import os
import sys
import time

from config_sim_analysis import networks, rdms_folder, similarity_folder, meg_rdm, meg_sensors
from bootstrapping import eval_bootstrap_pearson, get_rdms_vectors

sys.path.append('/home/hamza97/MFRS/')
from models.inception import inception_v3
from models.vgg import vgg16_bn
from models.mobilenet import mobilenet_v2
from models.cornet_s import cornet_s
from models.vgg import vgg16_bn
from models.resnet import resnet50
from models.FaceNet import FaceNet
from models.SphereFace import SphereFace

from utils.general import load_pickle, load_npy, save_npy, save_pickle



if __name__ == '__main__':
    start = time.time()

    N_bootstrap=100
    save = True

    meg_rdm=get_rdms_vectors(meg_rdm)
    #"inception_v3": inception_v3, "mobilenet": mobilenet_v2, "SphereFace": SphereFace, "resnet50": resnet50,
    networks_list = {
                "cornet_s": cornet_s, "FaceNet": FaceNet, "vgg16_bn": vgg16_bn}
    for model_name, model in networks_list.items():
        model_layers=networks[model_name]
        model_rdm = load_npy(os.path.join(rdms_folder, "%s_FamUnfam_data_rdm.npy"%model_name))
        model_rdm=get_rdms_vectors(model_rdm)
        network_layers = [item[0] for item in list(model(False, 1000, 1).named_modules())]
        sensor_r_list=eval_bootstrap_pearson(meg_rdm, model_rdm, network_layers, model_layers)
        if save:
            save_npy(sensor_r_list, os.path.join(similarity_folder, "%s_FamUnfam_bootstrap.npy"%model_name))
    time_sim = time.time() - start
    print('Computations ended in %s h %s m %s s' % (time_sim // 3600, (time_sim % 3600) // 60, time_sim % 60))
