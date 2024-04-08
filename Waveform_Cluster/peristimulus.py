import argparse
import zipfile
from glob import glob

import braingeneers.analysis as ba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from umap import UMAP


def load_curation(qm_path):
    with zipfile.ZipFile(qm_path, "r") as f_zip:
        qm = f_zip.open("qm.npz")
        data = np.load(qm, allow_pickle=True)
        spike_times = data["train"].item()
        fs = data["fs"]
        train = [times / fs for _, times in spike_times.items()]
        neuron_data = data["neuron_data"].item()
    return train, neuron_data


def load_data(
    exps,
    *,
    bin_size_ms,
    peristim_length_ms,
    fr_thresh,
    around_opto_end,
):
    all_data = {}

    # Load the experiment data and plot diagnostics if any of them are broken.
    for exp in exps:
        gpiofiles = glob(f"**/*{exp}*.npy", recursive=True)
        datafiles = glob(f"**/*{exp}*.zip", recursive=True)
        if len(gpiofiles) != 1:
            print(exp, "has", len(gpiofiles), "GPIO files.")
        if len(datafiles) == 0:
            print(exp, "has no data files.")
        elif len(datafiles) > 1:
            print(exp, "has ambiguous data:", *datafiles)
        if len(gpiofiles) == 1 and len(datafiles) == 1:
            train, neuron_data = load_curation(datafiles[0])
            all_data[exp] = dict(
                train=train, neuron_data=neuron_data, timestamps=np.load(gpiofiles[0])
            )

    # Add the peristimulus histogram and firing rates to the data.
    last_index = 0
    peristim_length_bins = round(peristim_length_ms / bin_size_ms)
    which_stamp = 1 if around_opto_end else 0
    for exp, data in all_data.items():
        sd = ba.SpikeData([t * 1e3 for t in data["train"]])
        frates = sd.rates("Hz")
        raster = sd.raster(bin_size_ms)
        fast_enough = frates >= fr_thresh
        data["unit_ids"] = np.arange(len(frates))[fast_enough]
        data["firing_rates"] = frates[fast_enough]
        raster = raster[fast_enough, :]
        center_bin = data["timestamps"][:, which_stamp] * 1e3 / bin_size_ms
        data["peristimulus"] = ph = np.zeros(
            (raster.shape[0], 2 * peristim_length_bins)
        )
        for bin in np.int32(np.round(center_bin)):
            startbin = bin - peristim_length_bins
            endbin = bin + peristim_length_bins
            if startbin < 0:
                chunk = raster[:, :endbin]
            else:
                chunk = raster[:, startbin:endbin]
            start = max(0, peristim_length_bins - bin)
            ph[:, start : start + chunk.shape[1]] += chunk
        ph /= len(center_bin)
        old_last_index = last_index
        last_index += ph.shape[0]
        data["index_range"] = (old_last_index, last_index)

    return all_data, peristim_length_bins


def normalized_psh(
    all_data, *, data_section, around_opto_end, peristim_length_bins, normalize
):
    # Gather the results into simple arrays.
    index_ranges = {}
    peristimulus = []
    total_fr = []
    for exp, data in all_data.items():
        peristimulus.append(data["peristimulus"])
        total_fr.append(data["firing_rates"])
        index_ranges[exp] = data["index_range"]
    peristimulus = np.vstack(peristimulus)
    total_fr = np.hstack(total_fr)

    # Select which part of the peristimulus to base normalization on.
    if data_section == "all":
        norm_base = peristimulus
    elif data_section not in ["opto", "control"]:
        raise ValueError(f"Unknown data section: {data_section}")
    else:
        use_start = around_opto_end == (data_section == "opto")
        if use_start:
            norm_base = peristimulus[:, :peristim_length_bins]
        else:
            norm_base = peristimulus[:, peristim_length_bins:]

    # Actually do the normalization, and return.
    norm = normalize(norm_base, 1, keepdims=True)
    norm[norm == 0] = 1
    return index_ranges, total_fr, peristimulus / norm, norm


def do_clustering(
    peristimulus,
    n_neighbors=10,
    min_dist=0.0,
    n_components=5,
    cluster_selection_epsilon=0.35,
    min_cluster_size=50,
    min_samples=None,
    alpha=1.0,
    seed=None,
):
    # Perform UMAP and HDBSCAN clustering.
    transformed = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=seed,
        n_jobs=-1 if seed is None else 1,
    ).fit_transform(peristimulus)
    clus = HDBSCAN(
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        alpha=alpha,
    ).fit(transformed)
    return transformed, clus.labels_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expfile", type=argparse.FileType("r"), help="File listing experiment names"
    )
    parser.add_argument(
        "--bin-size-ms", type=float, default=100, help="Bin size in milliseconds"
    )
    parser.add_argument(
        "--normalize-fr", action="store_true", help="Plot normalized firing rates."
    )
    parser.add_argument(
        "--around-opto-end", action="store_true", help="Use the end of the opto period."
    )
    parser.add_argument(
        "--peristim-length-ms",
        type=float,
        default=10e3,
        help="Length of the peristimulus histogram in milliseconds",
    )
    parser.add_argument(
        "--fr-thresh",
        type=float,
        default=0.25,
        help="Minimum firing rate to include a unit",
    )
    parser.add_argument(
        "--normalization-section",
        default="all",
        choices=["all", "opto", "control"],
        help="Which part of the PSH to use for normalization",
    )
    parser.add_argument(
        "--normalization",
        choices=["max", "mean", "median"],
        default="max",
        help="How to normalize the peristimulus histograms",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UMAP")
    args = parser.parse_args()

    with args.expfile as f:
        exps = f.read().splitlines()

    bin_size_sec = args.bin_size_ms / 1e3
    all_data, peristim_length_bins = load_data(
        exps,
        bin_size_ms=args.bin_size_ms,
        fr_thresh=args.fr_thresh,
        peristim_length_ms=args.peristim_length_ms,
        around_opto_end=args.around_opto_end,
    )
    index_ranges, total_fr, rescaled, norm = normalized_psh(
        all_data,
        around_opto_end=args.around_opto_end,
        data_section=args.normalization_section,
        peristim_length_bins=peristim_length_bins,
        normalize=getattr(np, args.normalization),
    )
    transformed, labels = do_clustering(rescaled, seed=args.seed)

    flat = PCA(n_components=2).fit_transform(transformed[labels >= 0, :])
    flat_labels = labels[labels >= 0]

    unique_labels = set(flat_labels)
    n_clusters = len(set(flat_labels))

    ax = plt.figure().gca()
    cluster_color = {
        label: f"C{label}" if label >= 0 else plt.cm.Spectral(each)
        for label, each in zip(unique_labels, np.linspace(0, 1, len(unique_labels)))
    }
    for k, col in cluster_color.items():
        class_index = np.nonzero(flat_labels == k)[0]
        for ci in class_index:
            ax.plot(
                flat[ci, 0],
                flat[ci, 1],
                "o",
                markerfacecolor=col,
                markeredgecolor="k",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    plt.savefig("clustered_umap.png")

    time = bin_size_sec * np.arange(rescaled.shape[1])
    ax = plt.figure().gca()
    for i in range(n_clusters):
        group = rescaled[labels == i, :]
        group_mean = group.mean(0)
        if not args.normalize_fr:
            group_mean *= norm[labels == i].mean() / bin_size_sec
        label = f"Cluster {i+1} ({group.shape[0]} Units)"
        ax.plot(time, group_mean, color=cluster_color[i], label=label)
        if args.normalize_fr:
            ax.set_ylabel("Normalized FR")
        else:
            ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlabel("Time (sec)")
    ax.legend()
    plt.savefig("cluster_means.png")

    print("\nKS Test of Mean FR Pre vs. Post:")
    psh = rescaled * norm
    f, axes = plt.subplots(n_clusters, 2)
    mean_pre = psh[:, :peristim_length_bins].mean(1) / bin_size_sec
    mean_post = psh[:, peristim_length_bins:].mean(1) / bin_size_sec
    for i in range(n_clusters):
        pre, post = mean_pre[labels == i], mean_post[labels == i]
        p = stats.ks_2samp(pre, post).pvalue
        sig = "*" if p < 0.01 / n_clusters else ""
        print(f"Cluster {i+1}: {p = :0.3} {sig}")
        maxrate = max(pre.max(), post.max())
        bins = np.linspace(0, maxrate, 35)
        axes[i, 0].hist(pre, color=f"C{i}", bins=bins)
        axes[i, 1].hist(post, color=f"C{i}", bins=bins)
        maxcount = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        for j in [0, 1]:
            axes[i, j].set_xlim(0, maxrate)
            axes[i, j].set_ylim(0, maxcount)
    axes[-1, 0].set_xlabel("Pre-Opto Firing Rate (Hz)")
    axes[-1, 1].set_xlabel("Post-Opto Firing Rate (Hz)")
    plt.savefig("histogram_fr.png")

    print("\nStandard Deviation:")
    f, axes = plt.subplots(n_clusters, 2)
    std_pre = psh[:, :peristim_length_bins].std(1) / bin_size_sec
    std_post = psh[:, peristim_length_bins:].std(1) / bin_size_sec
    for i in range(n_clusters):
        pre, post = std_pre[labels == i], std_post[labels == i]
        p = stats.ks_2samp(pre, post).pvalue
        sig = "*" if p < 0.01 / n_clusters else ""
        print(f"Cluster {i+1}: {p = :0.3} {sig}")
        maxrate = max(pre.max(), post.max())
        bins = np.linspace(0, maxrate, 35)
        axes[i, 0].hist(pre, color=f"C{i}", bins=bins)
        axes[i, 1].hist(post, color=f"C{i}", bins=bins)
        maxcount = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        for j in [0, 1]:
            axes[i, j].set_xlim(0, maxrate)
            axes[i, j].set_ylim(0, maxcount)
    axes[-1, 0].set_xlabel("Std. Dev. of Pre-Opto FR (Hz)")
    axes[-1, 1].set_xlabel("Std. Dev. of Post-Opto FR (Hz)")
    plt.savefig("histogram_fr_std.png")

    try:
        recording_slices = {
            slice: list(exps)
            for slice, exps in pd.read_csv("recording_slice_map.csv").groupby("sliceID").exp
        }

        f, axes = plt.subplots(len(recording_slices), 2)
        for i, (slice, exps) in enumerate(recording_slices.items()):
            units = np.zeros_like(labels, dtype=bool)
            for exp in exps:
                units[index_ranges[exp][0] : index_ranges[exp][1]] = True
            units &= labels < 0
            pre, post = mean_pre[units], mean_post[units]
            p = stats.ks_2samp(pre, post).pvalue
            sig = "*" if p < 0.01 / len(recording_slices) else ""
            print(f"Slice {slice}: {p = :0.3} {sig}")
            maxrate = max(pre.max(), post.max())
            bins = np.linspace(0, maxrate, 35)
            axes[i, 0].hist(pre, color=f"C{i}", bins=bins)
            axes[i, 1].hist(post, color=f"C{i}", bins=bins)
            maxcount = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
            for j in [0, 1]:
                axes[i, j].set_xlim(0, maxrate)
                axes[i, j].set_ylim(0, maxcount)
        axes[-1, 0].set_xlabel("Pre-Opto Firing Rate (Hz)")
        axes[-1, 1].set_xlabel("Post-Opto Firing Rate (Hz)")
        plt.savefig("histogram_fr_outlier_by_slice.png")

    except FileNotFoundError:
        print("No recording slice map found.")

    with open("used_recordings.txt", "w") as f:
        f.write("\n".join(all_data.keys()))
        f.write("\n")

    rows = []
    for exp, idces in index_ranges.items():
        for i in range(idces[0], idces[1]):
            rows.append(
                dict(
                    exp=exp,
                    unit=all_data[exp]["unit_ids"][i - idces[0]],
                    cluster=labels[i],
                    mean_pre=mean_pre[i],
                    mean_post=mean_post[i],
                    std_pre=std_pre[i],
                    std_post=std_post[i],
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv("cluster_assignment.csv", index=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
