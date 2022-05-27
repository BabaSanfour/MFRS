from extract_activations import model_activations, load_activations, activations_folder
from networks_rdm import networkRDM, load_rdms, rdms_dir
import sys
import os
import numpy as np
from similarity import subjects_similarity_score, stats_subjects_similarity_score, load_similarity_score, similarity_folder
sys.path.append('/home/hamza97/MFRS/')
from models.mobilenet import mobilenet_v2
from utils.load_data import Stimuliloader
import time
weights_path = '/home/hamza97/scratch/net_weights/'
study_path = '/home/hamza97/scratch/data/MFRS_data/'
meg_dir = '/home/hamza97/scratch/data/MFRS_data/MEG/'
meg_sensors = ['MEG0113',
 'MEG0112', 'MEG0111', 'MEG0122', 'MEG0123', 'MEG0121', 'MEG0132', 'MEG0133', 'MEG0131', 'MEG0143', 'MEG0142', 'MEG0141', 'MEG0213', 'MEG0212', 'MEG0211',
 'MEG0222', 'MEG0223', 'MEG0221', 'MEG0232', 'MEG0233', 'MEG0231', 'MEG0243', 'MEG0242', 'MEG0241', 'MEG0313', 'MEG0312', 'MEG0311', 'MEG0322', 'MEG0323',
 'MEG0321', 'MEG0333', 'MEG0332', 'MEG0331', 'MEG0343', 'MEG0342', 'MEG0341', 'MEG0413', 'MEG0412', 'MEG0411', 'MEG0422', 'MEG0423', 'MEG0421', 'MEG0432',
 'MEG0433', 'MEG0431', 'MEG0443', 'MEG0442', 'MEG0441', 'MEG0513', 'MEG0512', 'MEG0511', 'MEG0523', 'MEG0522', 'MEG0521', 'MEG0532', 'MEG0533', 'MEG0531',
 'MEG0542', 'MEG0543', 'MEG0541', 'MEG0613', 'MEG0612', 'MEG0611', 'MEG0622', 'MEG0623', 'MEG0621', 'MEG0633', 'MEG0632', 'MEG0631', 'MEG0642', 'MEG0643',
 'MEG0641', 'MEG0713', 'MEG0712', 'MEG0711', 'MEG0723', 'MEG0722', 'MEG0721', 'MEG0733', 'MEG0732', 'MEG0731', 'MEG0743', 'MEG0742', 'MEG0741', 'MEG0813',
 'MEG0812', 'MEG0811', 'MEG0822', 'MEG0823', 'MEG0821', 'MEG0913', 'MEG0912', 'MEG0911', 'MEG0923', 'MEG0922', 'MEG0921', 'MEG0932', 'MEG0933', 'MEG0931',
 'MEG0942', 'MEG0943', 'MEG0941', 'MEG1013', 'MEG1012', 'MEG1011', 'MEG1023', 'MEG1022', 'MEG1021', 'MEG1032', 'MEG1033', 'MEG1031', 'MEG1043', 'MEG1042',
 'MEG1041', 'MEG1112', 'MEG1113', 'MEG1111', 'MEG1123', 'MEG1122', 'MEG1121', 'MEG1133', 'MEG1132', 'MEG1131', 'MEG1142', 'MEG1143', 'MEG1141', 'MEG1213',
 'MEG1212', 'MEG1211', 'MEG1223', 'MEG1222', 'MEG1221', 'MEG1232', 'MEG1233', 'MEG1231', 'MEG1243', 'MEG1242', 'MEG1241', 'MEG1312', 'MEG1313', 'MEG1311',
 'MEG1323', 'MEG1322', 'MEG1321', 'MEG1333', 'MEG1332', 'MEG1331', 'MEG1342', 'MEG1343', 'MEG1341', 'MEG1412', 'MEG1413', 'MEG1411', 'MEG1423', 'MEG1422',
 'MEG1421', 'MEG1433', 'MEG1432', 'MEG1431', 'MEG1442', 'MEG1443', 'MEG1441', 'MEG1512', 'MEG1513', 'MEG1511', 'MEG1522', 'MEG1523', 'MEG1521', 'MEG1533',
 'MEG1532', 'MEG1531', 'MEG1543', 'MEG1542', 'MEG1541', 'MEG1613', 'MEG1612', 'MEG1611', 'MEG1622', 'MEG1623', 'MEG1621', 'MEG1632', 'MEG1633', 'MEG1631',
 'MEG1643', 'MEG1642', 'MEG1641', 'MEG1713', 'MEG1712', 'MEG1711', 'MEG1722', 'MEG1723', 'MEG1721', 'MEG1732', 'MEG1733', 'MEG1731', 'MEG1743', 'MEG1742',
 'MEG1741', 'MEG1813', 'MEG1812', 'MEG1811', 'MEG1822', 'MEG1823', 'MEG1821', 'MEG1832', 'MEG1833', 'MEG1831', 'MEG1843', 'MEG1842', 'MEG1841', 'MEG1912',
 'MEG1913', 'MEG1911', 'MEG1923', 'MEG1922', 'MEG1921', 'MEG1932', 'MEG1933', 'MEG1931', 'MEG1943', 'MEG1942', 'MEG1941', 'MEG2013', 'MEG2012', 'MEG2011',
 'MEG2023', 'MEG2022', 'MEG2021', 'MEG2032', 'MEG2033', 'MEG2031', 'MEG2042', 'MEG2043', 'MEG2041', 'MEG2113', 'MEG2112', 'MEG2111', 'MEG2122', 'MEG2123',
 'MEG2121', 'MEG2133', 'MEG2132', 'MEG2131', 'MEG2143', 'MEG2142', 'MEG2141', 'MEG2212', 'MEG2213', 'MEG2211', 'MEG2223', 'MEG2222', 'MEG2221', 'MEG2233',
 'MEG2232', 'MEG2231', 'MEG2242', 'MEG2243', 'MEG2241', 'MEG2312', 'MEG2313', 'MEG2311', 'MEG2323', 'MEG2322', 'MEG2321', 'MEG2332', 'MEG2333', 'MEG2331',
 'MEG2343', 'MEG2342', 'MEG2341', 'MEG2412', 'MEG2413', 'MEG2411', 'MEG2423', 'MEG2422', 'MEG2421', 'MEG2433', 'MEG2432', 'MEG2431', 'MEG2442', 'MEG2443',
 'MEG2441', 'MEG2512', 'MEG2513', 'MEG2511', 'MEG2522', 'MEG2523', 'MEG2521', 'MEG2533', 'MEG2532', 'MEG2531', 'MEG2543', 'MEG2542', 'MEG2541', 'MEG2612',
 'MEG2613', 'MEG2611', 'MEG2623', 'MEG2622', 'MEG2621', 'MEG2633', 'MEG2632', 'MEG2631', 'MEG2642', 'MEG2643', 'MEG2641']


if __name__ == '__main__':
    start = time.time()

    save = True
    # stimuli_hdf5_list = {"Fam": 150, "Unfam": 150, "Scram": 150,
    #         "FamUnfam": 300, "FamScram": 300, "UnfamScram": 300,
    #         "FamUnfamScram1": 300, "FamUnfamScram2": 300, "FamUnfamScram0": 450}
    stimuli_hdf5_list = {"FamUnfam": 300}

    networks_list = {"mobilenet": [mobilenet_v2, "mobilenet_0.01LR_32Batch_1000_VGGFace3"]}
    for model_name, model_param in networks_list.items():
        model = model_param[0](False, 1000, 1)
        weights = os.path.join(weights_path, model_param[1])
        for stimuli_file_name, cons in stimuli_hdf5_list.items():
            combinations_stats_file = os.path.join(similarity_folder, "%s_%s_data_stats.pkl"%(model_name, stimuli_file_name))
            if os.path.isfile(combinations_stats_file):
                print("combinations_stats file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                combinations_stats = load_similarity_score(combinations_stats_file)
            else:
                rdms_file = os.path.join(rdms_dir, "%s_%s_data_rdm.npy"%(model_name, stimuli_file_name))
                if os.path.isfile(rdms_file):
                    print("RDMs file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                    network_rdms = load_rdms(rdms_file)
                else:
                    images=Stimuliloader(cons, stimuli_file_name)
                    images = next(iter(images))
                    activations_file = os.path.join(activations_folder, "%s_%s_activations.pkl"%(stimuli_file_name, model_name))
                    if os.path.isfile(activations_file):
                        print("activations file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                        activations = load_activations(activations_file)
                    else:
                        activations=model_activations(model, images, weights, save = save, model_name = model_name, data_name = stimuli_file_name)

                    network_rdms=networkRDM(activations, cons, save = save, model_name = model_name, data_name = stimuli_file_name)
                subjects_sim_dict_file = os.path.join(similarity_folder, "%s_%s_data_sim_scores.pkl"%(model_name, stimuli_file_name))
                if os.path.isfile(subjects_sim_dict_file):
                    print("subjects sim file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                    subjects_sim_dict = load_similarity_score(subjects_sim_dict_file)
                else:
                    out_file = os.path.join(meg_dir, "RDMs_16-subject_1-sub_opt1-chl_opt.npy")
                    meg_rdms = np.load(out_file)
                    network_layers = [item[0] for item in list(model.named_modules())]
                    subjects_sim_dict = subjects_similarity_score(meg_rdms, meg_sensors, network_rdms, network_layers, save_subjects = save,
                                    save_subject = save, model_name = model_name, data_name = stimuli_file_name)
                combinations_stats=stats_subjects_similarity_score(subjects_sim_dict, save = save, model_name = model_name, data_name = stimuli_file_name)
    time_sim = time.time() - start
    print('Computations ended in %s h %s m %s s' % (time_sim // 3600, (time_sim % 3600) // 60, time_sim % 60))
