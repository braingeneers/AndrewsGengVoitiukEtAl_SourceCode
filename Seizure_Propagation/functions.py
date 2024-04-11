import braingeneers.utils.smart_open_braingeneers as smart_open
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import utils as utils


def read_train(qm_path):
    with smart_open.open(qm_path, 'rb') as f:
        with zipfile.ZipFile(f, 'r') as f_zip:
            qm = f_zip.open("qm.npz")
            data = np.load(qm, allow_pickle=True)
            spike_times = data["train"].item()
            fs = data["fs"]
            train = [times / fs for __, times in spike_times.items()]
    return train


def load_curation(qm_path):
    with smart_open.open(qm_path, 'rb') as f:
        with zipfile.ZipFile(f, 'r') as f_zip:
            qm = f_zip.open("qm.npz")
            data = np.load(qm, allow_pickle=True)
            spike_times = data["train"].item()
            fs = data["fs"]
            train = [times / fs for _, times in spike_times.items()]
            if "config" in data:
                config = data["config"].item()
            else:
                config = None
            neuron_data = data["neuron_data"].item()
    return train, neuron_data, config, fs


def plot_inset(axs, temp_pos, templates, nelec=2, ylim_margin=0, pitch=17.5):
    assert len(temp_pos) == len(templates), "Input length must be the same!"
    # find the max template
    if isinstance(templates, list):
        templates = np.asarray(templates)
    amp = np.max(templates, axis=1) - np.min(templates, axis=1)
    max_amp_index = np.argmax(amp)
    position = temp_pos[max_amp_index]
    axs.scatter(position[0], position[1], linewidth=10, alpha=0.2, color='grey')
    axs.text(position[0], position[1], str(position), color="g", fontsize=12)
    # set same scaling to the insets
    ylim_min = min(templates[max_amp_index])
    ylim_max = max(templates[max_amp_index])
    # choose channels that are close to the center channel
    for i in range(len(temp_pos)):
        chn_pos = temp_pos[i]
        if position[0] - nelec * pitch <= chn_pos[0] <= position[0] + nelec * pitch \
                and position[1] - nelec * pitch <= chn_pos[1] <= position[1] + nelec * pitch:
            axin = axs.inset_axes([chn_pos[0]-5, chn_pos[1]-5, 15, 20], transform=axs.transData)
            axin.plot(templates[i], color='k', linewidth=2, alpha=0.7)
            axin.set_ylim([ylim_min - ylim_margin, ylim_max + ylim_margin])
            axin.set_axis_off()
    axs.set_xlim(position[0]-1.5*nelec*pitch, position[0]+1.5*nelec*pitch)
    axs.set_ylim(position[1]-1.5*nelec*pitch, position[1]+1.5*nelec*pitch)
    axs.invert_yaxis()
    return axs


def plot_unit_footprint(qm_path, title="", save_to=None):
    """
    plot footprints for all units in one figure
    """
    _, neuron_data, _, _ = load_curation(qm_path)

    for k, data in neuron_data.items():
        cluster = data["cluster_id"]
        npos = data["neighbor_positions"]
        ntemp = data["neighbor_templates"]

        fig, axs = plt.subplots(figsize=(4, 4))
        axs = plot_inset(axs=axs, temp_pos=npos, templates=ntemp)
        axs.set_title(f"{title} Unit {cluster} ")
        if save_to is not None:
            plt.savefig(f"{save_to}/footprint_{title}_unit_{cluster}.png", dpi=300)
            plt.close()


def plot_raster_with_fr(train:list, title:list, bin_size=0.1, w=5, avg=False, axs=None):
    bins, fr_avg = utils.get_population_fr(trains=train, bin_size=bin_size, w=w, average=avg)
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(16, 6))
    axs.set_title(f"raster_{title}", fontsize=12)

    y = 0
    for vv in train:
        axs.scatter(vv, [y]*len(vv), marker="|", c='k', s=4, alpha=0.7)
        y += 1
    axs.set_xlabel("Time (s)", fontsize=16)
    axs.set_ylabel("Unit", fontsize=16)
    axs.xaxis.set_tick_params(labelsize=16)
    axs.yaxis.set_tick_params(labelsize=16)
    
    axs1 = axs.twinx()
    axs1.yaxis.set_label_position("right") 
    axs1.spines['right'].set_color('r')
    axs1.spines['right'].set_linewidth(3)
    axs1.plot(bins[1:], fr_avg, color='r', linewidth=3, alpha=0.5)
    axs1.set_ylabel("Population Firing Rate (Hz)", fontsize=16, color='r')
    axs1.set_xlabel("Time (s)", fontsize=16)
    axs1.yaxis.set_tick_params(labelsize=16)
    axs1.spines['top'].set_visible(False)
    axs1.get_xaxis().set_visible(False)
    axs1.tick_params(left=False, right=True, labelleft=False, labelright=True,
                    bottom=False, labelbottom=True)
    axs1.tick_params(axis='y', colors='r')

    return axs