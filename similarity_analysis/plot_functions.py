import os
import mne
import matplotlib.pyplot as plt
from config_sim_analysis import sensors_position, plot_folder, mask_params
from plot_utils import match_layers_sensor, get_network_layers_info
from similarity import get_main_network_similarity_scores


def plot_MEG_topomaps(similarity_values: dict, extremum: int, axes, i: int, ylabel: str, min_fig: plt.Figure, last: bool = False, sensors_position: mne.Info = sensors_position):
    """Plot the 3 topomaps (MAG, Grad1, Grad2) for a single layer/whole network similarity scores."""
    im, _ = mne.viz.plot_topomap(similarity_values[0], sensors_position, show=False, vmax=extremum[0],
                                 vmin=-extremum[0], axes=axes[i][0], extrapolate='head')
    axes[i][0].set_ylabel(ylabel, fontweight ='bold', fontsize = 10)
    im1, _ = mne.viz.plot_topomap(similarity_values[1], sensors_position, show=False, vmax=extremum[1],
                                  vmin=-extremum[1], axes=axes[i][1], extrapolate='head')
    im2, _ = mne.viz.plot_topomap(similarity_values[2], sensors_position, show=False, vmax=extremum[2],
                                  vmin=-extremum[2], axes=axes[i][2], extrapolate='head')
    if last:
        min_fig.colorbar(im, ax=axes[i][0], orientation='horizontal')
        axes[i][0].set_xlabel("MAG", fontweight ='bold', fontsize = 10)
        min_fig.colorbar(im1, ax=axes[i][1], orientation='horizontal')
        axes[i][1].set_xlabel("Grad1", fontweight ='bold', fontsize = 10)
        min_fig.colorbar(im2, ax=axes[i][2], orientation='horizontal')
        axes[i][2].set_xlabel("Grad2", fontweight ='bold', fontsize = 10)

def plot_similarity(similarity_scores, extremum, save=False, network_name=None, correlation='pearson'):
    """Plot the 3 topomaps (MAG, Grad1, Grad2) for a single layer/whole network similarity scores."""
    length = len(similarity_scores)
    fig, axes = plt.subplots(length, 3, figsize=(10, length*3))
    i, last = 0, False
    for name, similarity_values in similarity_scores.items():
        if i == length-1:
            last = True
        plot_MEG_topomaps(similarity_values, extremum, axes, i, name, fig, last)
        i+=1
    if save:
        fig.show()
        fig.savefig(os.path.join(plot_folder, '%s_%s_all_layers_similiarity_topomaps.png'%(network_name, correlation)))

def plot_layers_similarity_bars(networks, channels_list, sensor_type, mask_params=mask_params,  save=True):
    """Plot the bars of the maximum similarity values for each layer for the provided networks + the topomap of the layer with highest similarity value."""
    fig, axes = plt.subplots(len(networks), 2, figsize=(12, len(networks)*3), gridspec_kw={'width_ratios': [2, 1]})
    for i, network in enumerate(networks.keys()):
        network_layers=networks[network]

        main_similarity_scores=get_main_network_similarity_scores('%s_FamUnfam_data_sim_scores'%network, network_layers)
        correlations_list, sensors_list = match_layers_sensor(main_similarity_scores, network_layers, channels_list)
        idx, best_layer, sim_chls, extremum, mask = get_network_layers_info(correlations_list,
                                                                       network_layers, main_similarity_scores, sensors_list, channels_list)

        barlist=axes[i][0].bar(networks[network], correlations_list, width=0.4, color=(0.2, 0.4, 0.8, 0.9))
        axes[i][0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[i][0].text(x=0.1 , y =0.022 , s="Input", fontdict=dict(fontsize=10))
        axes[i][0].text(x=len(network_layers)-3, y =0.022 , s="Output", fontdict=dict(fontsize=10))
        if sensor_type=="MAG":
            axes[i][0].set_ylim([0.03, 0.09])
        else:
            axes[i][0].set_ylim([0.03, 0.12])
        axes[i][0].set_xlabel('%s Layers'%network, fontweight ='bold', fontsize = 10)
        axes[i][0].scatter(idx,max(correlations_list)+0.002, marker="*", c="white", edgecolors="r")

        im,cm=mne.viz.plot_topomap(sim_chls, sensors_position, show=False, vmax=extremum,
                                   vmin=-extremum, axes=axes[i][1], mask=mask, mask_params=mask_params,
                                   extrapolate='head')

        fig.colorbar(im, ax=axes[i][1])
        axes[i][1].set_xlabel("%s %sth Layer"%(network, str(idx)), fontweight ='bold', fontsize = 10)
        if int(len(networks)/2)==i:
            axes[i][0].set_ylabel('Pearson correlation values', fontweight ='bold', fontsize = 10)

        axes[i][0].spines['right'].set_visible(False)
        axes[i][0].spines['top'].set_visible(False)

    fig.text(x=0.09 , y =0.97 , s="A", fontdict=dict(fontsize=15, fontweight ='bold'))
    fig.text(x=0.62 , y =0.97 , s="B", fontdict=dict(fontsize=15, fontweight ='bold'))

    if save:
        fig.show()
        fig.savefig(os.path.join(plot_folder, '%s_layers_similiarity_bar.png'%(sensor_type)))

def plot_networks_results(models, extremum_3, max_sim, accuracy, params, name="Networks", save=True, correlation_measure="Pearson"):
    """plot the networks highest similarity value for each sensor type,
        the decoding top5 accuracy, and the 3 topomaps for the networks similarity results. ."""
    fig = plt.figure(constrained_layout=True, figsize=(len(models)*2.6, len(models)*3.3))
    subfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=params["height_ratios"],  wspace=0.07)
    figtop = subfigs[0]
    ax1, ax2 =figtop.subplots(1,2,gridspec_kw={'width_ratios': [3, 1]})
    barWidth = 0.2
    br1 = np.arange(len(models))
    br2 = [x + barWidth  for x in range(len(models))]
    br3 = [x + 2*barWidth  for x in range(len(models))]
    # Make the plot
    ax1.bar(br1, max_sim[0], color ='#a5d5d8', width = barWidth,
            edgecolor ='grey', label ='MAG')
    ax1.bar(br2, max_sim[1], color ='#73a2c6', width = barWidth,
            edgecolor ='grey', label ='Grad1')
    ax1.bar(br3, max_sim[2], color ='#00429d', width = barWidth,
            edgecolor ='grey', label ='Grad2')
    ax1.set_ylim([0.03, 0.1])

    ax2.bar(br1, accuracy, color ='#ffffe0', width = 0.4,
            edgecolor ='grey', label ='Accuracy')
    ax2.tick_params(labelrotation=25)
    ax1.tick_params(labelrotation=25)
    ax2.set_ylim([50, 100])
    # Adding Xticks
    ax1.set_ylabel('%s correlations'%correlation_measure, fontweight ='bold', fontsize = 10)
    ax1.set_xticks([r + barWidth for r in range(len(max_sim[0]))],
            models.keys())

    ax1.legend(loc='upper left', prop={'size': 8})
    ax2.legend(prop={'size': 8})
    ax2.set_ylabel('Top5 Accuracy',fontweight ='bold', fontsize = 10)
    ax2.set_xticks([r  for r in range(len(max_sim[0]))],
           models.keys())
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    figDown = subfigs[1]

    axes= figDown.subplots(len(models),3)
    for i, network in enumerate(models.keys()):
        last = i == len(models)-1
        plot_MEG_topomaps(models[network], extremum_3, axes, i, network, figDown, last)
        axes[i][0].set_ylabel(network, fontweight ='bold', fontsize = 10)


    fig.text(x=0 , y = params["AB"], s="A", fontdict=dict(fontsize=15, fontweight ='bold'))
    fig.text(x=0.7 , y = params["AB"] , s="B", fontdict=dict(fontsize=15, fontweight ='bold'))

    fig.text(x=0 , y = params["C"] , s="C", fontdict=dict(fontsize=15, fontweight ='bold'))


    if save:
        fig.show()
        fig.savefig(os.path.join(plot_folder, '%s_overall_results.png'%name))
