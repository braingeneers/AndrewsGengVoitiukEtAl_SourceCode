import numpy as np
import io
import zipfile
import pandas as pd

def sort_template_amplitude(template):
    """
    sort template by amplitude from the largest to the smallest
    :param template: N x M array template array as N for the length of samples,
                     and M for the length of channels
    :return: sorted template index
    """
    assert template.ndim == 2, "Input should be a 2D array; use sort_templates() for higher dimensional data"
    amp = np.max(template, axis=0) - np.min(template, axis=0)
    sorted_idx = np.argsort(amp)[::-1]
    return sorted_idx


def get_best_channel(channels, template):
    assert len(channels) == template.shape[1], "The number of channels does not match to template"
    idx = sort_template_amplitude(template)
    return channels[idx[0]]


def get_best_channel_cluster(clusters, channels, templates):
    """
    find the best channel by sorting templates by amplitude.
    :param clusters:
    :param channels:
    :param templates:
    :return:
    """
    assert len(clusters) == len(templates), "The number of clusters not equal to the number of templates"
    best_channel = dict.fromkeys(clusters)
    for i in range(len(clusters)):
        cls = clusters[i]
        temp = templates[i]
        best_channel[cls] = get_best_channel(channels, temp)
    return best_channel
