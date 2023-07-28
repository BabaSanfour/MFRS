import os
import mne
import matplotlib.pyplot as plt
import numpy as np

from utils.config import sensors_position, plots_path
from matplotlib.lines import Line2D


def plot_MEG_topomaps(similarity_values: list, extremum_values: list, axes: plt, i: int, ylabel: str, min_fig: plt.Figure, first: bool = False, sensors_position: mne.Info = sensors_position):
    """
    Plot the three topomaps (MAG, Grad1, Grad2) for a single layer.
    
    Args:
    - similarity_values (list): List containing the similarity values for each sensor type.
    - vlim (tuple): Colormap limits to use. If a tuple of floats, specifies the lower and upper bounds of the colormap (in that order).
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
    im, _ = mne.viz.plot_topomap(similarity_values[0], sensors_position, show=False, vlim=extremum_values,
                                 sphere=0.18, axes=axes[i][0], extrapolate='head')
    axes[i][0].set_ylabel(ylabel, fontweight='bold', fontsize=14)

    # Plot Grad1 topomap
    im1, _ = mne.viz.plot_topomap(similarity_values[1], sensors_position, show=False, vlim=extremum_values,
                                   sphere=0.18, axes=axes[i][1], extrapolate='head')

    # Plot Grad2 topomap
    im2, _ = mne.viz.plot_topomap(similarity_values[2], sensors_position, show=False, vlim=extremum_values,
                                  sphere=0.18, axes=axes[i][2], extrapolate='head')

    # Add colorbar 
    min_fig.colorbar(im, ax=axes[i][2], orientation='vertical')
    if first:
        axes[i][0].set_xlabel("MAG", fontweight='bold', fontsize=16)
        axes[i][1].set_xlabel("Grad1", fontweight='bold', fontsize=16)
        axes[i][2].set_xlabel("Grad2", fontweight='bold', fontsize=16)

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
        file_path = os.path.join(plots_path, file_name)
        fig.savefig(file_path)


def plot_single_network(ax, network_layers, values_dict, max_index_list, network, errors, upper_errors_list, lower_errors_list, colors_list, linestyle_list, noise_ceiling, noise_ceiling_values: list = [0.0176, 0.2586]):
    """
    Plot the bars of the maximum similarity values for a single network.

    Args:
    - ax: The axes object for the subplot.
    - network_layers (list): List of network layers.
    - values_dict (dict): Dictionary mapping stimuli file names to similarity values.
    - max_index_list (list): List of indices of the layers with the highest similarity for each stimuli file.
    - network (str): The name of the network.
    - errors (bool): Flag indicating whether to show error bars.
    - upper_errors_list (list): List of upper error values for each stimuli file.
    - lower_errors_list (list): List of lower error values for each stimuli file.
    - colors_list (list): List of colors for each stimuli file.
    - linestyle_list (list): List of line styles for each stimuli file.
    - noise_ceiling (bool): Flag indicating whether to show the noise ceiling.
    - noise_ceiling_values (list, optional): List of lower and upper noise ceiling values. Default is [0.0176, 0.2586].

    Returns:
    - None
    """
    x = np.arange(len(network_layers))
    smooth_x = np.linspace(x.min(), x.max(), 300)  # Smooth x-values for interpolation
    custom_lines = []
    legend = []
    for i, (Stimuli_file_name, values_list) in enumerate(values_dict.items()):
        smooth_values = np.interp(smooth_x, x, values_list)  # Smoothed values using cubic interpolation
        ax.plot(smooth_x, smooth_values, color=colors_list[i])
        ax.scatter(max_index_list[i], max(values_list) + 0.002, marker="*", c="white", edgecolors="r")
        if errors:
            ax.fill_between(x, values_list - lower_errors_list[i], values_list + upper_errors_list, alpha=0.2, color=colors_list[i])
        custom_lines.append(Line2D([0], [0], color=colors_list[i], linestyle=linestyle_list[i]))
        legend.append(Stimuli_file_name)
    
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labelrotation=45)
    ax.set_xticks(x[::5])
    ax.text(x=0.1, y=0, s="Input", fontdict=dict(fontsize=10))
    ax.text(x=len(network_layers)-3, y=0, s="Output", fontdict=dict(fontsize=10))
    if noise_ceiling:
        ax.axhline(y=noise_ceiling_values[0], linewidth=2, color='#e0bb41')
        custom_lines.append(Line2D([0], [0], color='#e0bb41', lw=2))
        legend.append('Lower Noise Ceiling')
        ax.axhline(y=noise_ceiling_values[1], linewidth=2, color='#873e23')
        custom_lines.append(Line2D([0], [0], color='#873e23', lw=2))
        legend.append('Upper Noise Ceiling')
        max_value = np.max([np.max(values) for values in values_dict.values()]) + 0.03
        plt.ylim(0, max_value)  # Set the lower limit to 0 and upper limit to max_value
        plt.ylim(noise_ceiling_values[1]-0.01, noise_ceiling_values[1]+0.01, ymin=0)

    ax.set_xlabel(f'{network} Layers', fontweight='bold', fontsize=10)
    ax.legend(custom_lines, legend)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_single_network_topomap(ax, fig, filtered_values: dict, extremum: int, max_layer_name: str, network: str, mask, mask_params):
    """
    Plot the topomap for a single network.
    Args:
    - ax: The axes object for the subplot.
    - fig: The figure object for color bars.
    - filtered_values (dict): Dictionary containing the similarity values for each sensor type.
    - extremum (int): Maximum absolute value for this sensor type.
    - max_layer_name (str): Name of the layer with the highest similarity.
    - network: Name of network.
    - mask: The sensor mask.
    - mask_params: Parameters for creating the sensor mask.

    Returns:
    - None
    """
    im, cm = mne.viz.plot_topomap(filtered_values, sensors_position, show=False, vmax=extremum,
                                vmin=-extremum, axes=ax, sphere=0.18, mask=mask, mask_params=mask_params,
                                extrapolate='head')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(f"{network} {max_layer_name} Layer", fontweight='bold', fontsize=10)

