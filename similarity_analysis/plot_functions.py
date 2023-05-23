import os
import mne
import matplotlib.pyplot as plt
import numpy as np

from utils.config_sim_analysis import sensors_position, plot_folder, mask_params
from plot_utils import match_layers_sensor, get_network_layers_info, get_bootstrap_values
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


def plot_MEG_topomaps(similarity_values: dict, extremum_values: list, axes: plt, i: int, ylabel: str, min_fig: plt.Figure, last: bool = False, sensors_position: mne.Info = sensors_position):
    """
    Plot the three topomaps (MAG, Grad1, Grad2) for a single layer or whole network similarity scores.
    
    Args:
    - similarity_values (dict): Dictionary containing the similarity values for each sensor type.
    - extremum_values (list): List containing the maximum absolute value for each sensor type.
    - axes: The axes object for the subplot grid.
    - i (int): The index of the current subplot.
    - ylabel (str): The label for the y-axis.
    - min_fig: The parent figure object.
    - last (bool): Flag indicating if it is the last subplot. Default is False.
    - sensors_position: The position information of the sensors. Default is sensors_position.

    Returns:
    - None
    """

    # Plot MAG topomap
    im, _ = mne.viz.plot_topomap(similarity_values[0], sensors_position, show=False, vmax=extremum_values[0],
                                 vmin=-extremum_values[0],  sphere=0.18, axes=axes[i][0], extrapolate='head')
    axes[i][0].set_ylabel(ylabel, fontweight='bold', fontsize=10)

    # Plot Grad1 topomap
    im1, _ = mne.viz.plot_topomap(similarity_values[1], sensors_position, show=False, vmax=extremum_values[1],
                                  vmin=-extremum_values[1], sphere=0.18, axes=axes[i][1], extrapolate='head')

    # Plot Grad2 topomap
    im2, _ = mne.viz.plot_topomap(similarity_values[2], sensors_position, show=False, vmax=extremum_values[2],
                                  vmin=-extremum_values[2], sphere=0.18, axes=axes[i][2], extrapolate='head')

    if last:
        # Add colorbar for MAG topomap
        min_fig.colorbar(im, ax=axes[i][0], orientation='horizontal')
        axes[i][0].set_xlabel("MAG", fontweight='bold', fontsize=10)

        # Add colorbar for Grad1 topomap
        min_fig.colorbar(im1, ax=axes[i][1], orientation='horizontal')
        axes[i][1].set_xlabel("Grad1", fontweight='bold', fontsize=10)

        # Add colorbar for Grad2 topomap
        min_fig.colorbar(im2, ax=axes[i][2], orientation='horizontal')
        axes[i][2].set_xlabel("Grad2", fontweight='bold', fontsize=10)


def plot_similarity(similarity_scores: dict, extremum_values: list, network_name: str, stimuli_file_name: str, save: bool = False, correlation: str = 'pearson'):
    """
    Plot the three topomaps (MAG, Grad1, Grad2) for each layer or the whole network similarity scores.

    Args:
    - similarity_scores (dict): Dictionary containing the similarity scores for each layer.
    - extremum_values (list): List containing the maximum absolute value for each sensor type.
    - save (bool): Flag indicating whether to save the figure. Default is False.
    - network_name (str): Name of the network. Default is None.
    - correlation (str): Type of correlation measure used. Default is 'spearman'.

    Returns:
    - None
    """

    fig, axes = plt.subplots(len(similarity_scores), 3, figsize=(10, len(similarity_scores) * 3))

    for i, (name, similarity_values) in enumerate(similarity_scores.items()):
        last = i == len(similarity_scores) - 1

        plot_MEG_topomaps(similarity_values, extremum_values, axes, i, name, fig, last)

    if save:
        plt.show()
        file_name = f"{network_name}_{stimuli_file_name}_{correlation}_all_layers_similarity_topomaps.png"
        file_path = os.path.join(plot_folder, file_name)
        fig.savefig(file_path)


def plot_layers_similarity_bars(networks, channels_list, sensor_type, file="FamUnfam",correlation_measure="spearman", avg=False,
                                show_bootstrap_bars=False, percentile=5, mask_params=mask_params,  save=True):
    """Plot the bars of the maximum similarity values for each layer for the provided networks + the topomap of the layer with highest similarity value."""
    fig, axes = plt.subplots(len(networks), 2, figsize=(12, len(networks)*3), gridspec_kw={'width_ratios': [2, 1]})
    for i, network in enumerate(networks.keys()):
        network_layers=networks[network]
        if avg:
            main_similarity_scores=get_main_network_similarity_scores('%s_%s_data_sim_scores_avg'%(network, file), network_layers)
        else:
            main_similarity_scores=get_main_network_similarity_scores('%s_%s_data_sim_scores'%(network, file), network_layers)

        correlations_list, sensors_list = match_layers_sensor(main_similarity_scores, network_layers, channels_list, correlation_measure)
        idx, best_layer, sim_chls, extremum, mask = get_network_layers_info(correlations_list,
                                                                       network_layers, main_similarity_scores, sensors_list, channels_list)
        if show_bootstrap_bars:
            bootstrap_error_values=get_bootstrap_values(network, sensors_list, percentile)
            # axes[i][0].errorbar(x=networks[network], y=correlations_list,  yerr=bootstrap_error_values, elinewidth=0.8)
            barlist=axes[i][0].bar(networks[network], correlations_list, yerr=bootstrap_error_values,
                                    width=0.4, color=(0.2, 0.4, 0.8, 0.9))

        else:
            barlist=axes[i][0].bar(networks[network], correlations_list,
                                    width=0.4, color=(0.2, 0.4, 0.8, 0.9))
        axes[i][0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[i][0].text(x=0.1 , y =-0.03 , s="Input", fontdict=dict(fontsize=10))
        axes[i][0].text(x=len(network_layers)-3, y =-0.03 , s="Output", fontdict=dict(fontsize=10))
        # if sensor_type=="MAG":
        #     axes[i][0].set_ylim([0, 0.27])
        # else:
        #     axes[i][0].set_ylim([0, 0.27])
        # axes[i][0].axhline(y=0.2586,linewidth=2, color='#873e23')
        axes[i][0].axhline(y=0.0176,linewidth=2, color='#e0bb41')

        axes[i][0].set_xlabel('%s Layers'%network, fontweight ='bold', fontsize = 10)
        axes[i][0].scatter(idx,max(correlations_list)+0.002, marker="*", c="white", edgecolors="r")
        custom_lines = [Line2D([0], [0], color='#873e23', lw=2),
                Line2D([0], [0], color='#e0bb41', lw=2)]
        axes[i][0].legend(custom_lines, ['Upper Noise Ceiling', 'Lower Noise Ceiling'])


        im,cm=mne.viz.plot_topomap(sim_chls, sensors_position, show=False, vmax=extremum,
                                   vmin=-extremum, axes=axes[i][1], mask=mask, mask_params=mask_params,
                                   extrapolate='head')

        fig.colorbar(im, ax=axes[i][1])
        axes[i][1].set_xlabel("%s %sth Layer"%(network, str(idx+1)), fontweight ='bold', fontsize = 10)
        if int(len(networks)/2)==i:
            axes[i][0].set_ylabel('Spearman correlation values', fontweight ='bold', fontsize = 10)

        axes[i][0].spines['right'].set_visible(False)
        axes[i][0].spines['top'].set_visible(False)

    fig.text(x=0.09 , y =0.97 , s="A", fontdict=dict(fontsize=15, fontweight ='bold'))
    fig.text(x=0.62 , y =0.97 , s="B", fontdict=dict(fontsize=15, fontweight ='bold'))

    if save:
        fig.show()
        fig.savefig(os.path.join(plot_folder, '%s_layers_similiarity_bar.png'%(sensor_type)))

def plot_networks_results(models, extremum_3, max_sim, accuracy, params, name="Networks", save=True, correlation_measure="spearman"):
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
    ax1.set_ylim([0, 0.27])
    ax3 = ax1.twinx()

    ax1.axhline(y=0.2586,linewidth=2, color='#873e23')
    ax1.axhline(y=0.0176,linewidth=2, color='#e0bb41')
    custom_lines = [Line2D([0], [0], color='#873e23', lw=2),
            Line2D([0], [0], color='#e0bb41', lw=2)]

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
    ax3.legend(custom_lines, ['Upper Noise Ceiling', 'Lower Noise Ceiling'])

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
