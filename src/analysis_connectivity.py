import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
from collections import OrderedDict

from src.cortical_embedding import get_distance_from


"""
connectivity matrices
1) Trained connectivity matrix raw
2) Trained connectivity matrix with duplicates summed over
3) FLN connectivity matrix with FLN_ij and nonzero indices

weight distributions 
4) weight magnitude over distance with exponential fit
5) ... 

circle plot for visualising the gradient of connectivity 
6) circle plot of raw connecivity duplicates summed over 

extra functions to help understand the connectivity
- distirbution of weights to see the magnitude (sanity check for nonzero threshold)

"""


def sum_over_duplicates(connectivity_matrix: np.ndarray, ce, mode="sum") -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, 180] by averaging 'duplicate' sets
    of neurons that belong to each cortical area.
    """
    cortical_areas = ce.cortical_areas
    num_units_new = len(cortical_areas)
    num_units_old = connectivity_matrix.shape[0]
    connectivity_matrix_rows_summed = np.zeros(
        (num_units_new, num_units_old), dtype=float
    )
    connectivity_matrix_summed = np.zeros((num_units_new, num_units_new), dtype=float)

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:  # if area has duplicates
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                connectivity_matrix_rows_summed[i, :] = connectivity_matrix[
                    start_idx : start_idx + dup_count, :
                ].sum(axis=0)
                if mode == "mean":
                    connectivity_matrix_rows_summed[i, :] /= dup_count

        else:
            connectivity_matrix_rows_summed[i, :] = connectivity_matrix[i, :]

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                # print(f"area {area} has {dup_count} duplicates")
                # print(connectivity_matrix_summed[:, i].shape)
                # print(
                #     connectivity_matrix[:, start_idx : start_idx + dup_count]
                #     .sum(axis=0)
                #     .shape
                # )
                connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[
                    :, start_idx : start_idx + dup_count
                ].sum(axis=1)
                if mode == "mean":
                    connectivity_matrix_summed[:, i] /= dup_count
        else:
            connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[:, i]
    return connectivity_matrix_summed


def fig_3_weights_over_distance_lambda_fitted(model, thres=0.001, provided_weiths=None):
    if provided_weiths is not None:
        weights = provided_weiths.flatten()
    else:
        weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy().flatten()

    nonzero_ind, nonzero_weights_flattened = remove_near_zero(weights, thres)
    density = calculate_densitiy(weights.shape[0], nonzero_ind)
    print(f"Density: {density:.2f}%")
    distances = model.ce.distance_matrix.flatten()[nonzero_ind]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        distances,
        abs(nonzero_weights_flattened),
        c="blue",
        alpha=0.5,
        marker="o",
    )
    ax.set_ylabel("Absolute Weight")
    ax.set_xlabel("Distance")
    ax.set_title(
        f"Weights Over Distance"
    )
    # Calculate spearman correlation
    correlation, p_value = spearmanr(distances, abs(nonzero_weights_flattened))
    print(f"Spearman correlation: {correlation}, p-value: {p_value}")
    # Fit an exponential distribution to the total weights
    valid_indices = nonzero_weights_flattened > 0  # Exclude bins with zero total weight
    x_data = distances[valid_indices]
    y_data = abs(nonzero_weights_flattened[valid_indices])
    params, _ = curve_fit(exponential, x_data, y_data, p0=(1, 0.1))
    a, lambda_ = params
    print(f"Fitted exponential parameters: a={a}, lambda={lambda_}")
    # Plot the fitted exponential curve
    x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
    y_fit = exponential(x_fit, a, lambda_)
    ax.plot(x_fit, y_fit, color="red", label="lambda value: {:.2f}".format(lambda_))
    
    ax.legend()

    return fig, ax, density, lambda_


def fig_3_histogram_weights_over_distance_lambda(model, thres=0.001, bins=50, p=False):
    trained_weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy().flatten()
    full_num_weights = trained_weights.flatten().shape[0]

    nonzero_ind, nonzero_weights_flattened = remove_near_zero(trained_weights, thres)
    density = calculate_densitiy(full_num_weights, nonzero_ind)
    print(f"Density: {density:.2f}%")
    distance_matrix = model.ce.distance_matrix.flatten()[nonzero_ind]
    # distance_matrix = distance_matrix.flatten()[nonzero_ind]

    # Bin distances and calculate total weights
    distance_bins = np.linspace(0, np.max(distance_matrix) + 1, bins)
    bin_indices = np.digitize(distance_matrix, distance_bins)
    total_weights = np.zeros(len(distance_bins) - 1)

    if p == True:
        # Calculate the density of weights in each bin
        density, _ = np.histogram(distance_matrix, bins=distance_bins)
        density = density / np.sum(density)

    for i in range(len(total_weights)):
        bin_mask = bin_indices == i
        total_weights[i] = np.sum(nonzero_weights_flattened[bin_mask])

    # Plot the bar plot
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        bin_centers,
        abs(total_weights),
        width=(distance_bins[1] - distance_bins[0]),
        alpha=0.7,
        label="Total Weight",
    )

    # Fit an exponential distribution to the total weights
    valid_indices = total_weights > 0  # Exclude bins with zero total weight
    x_data = bin_centers[valid_indices]
    y_data = total_weights[valid_indices]

    params, _ = curve_fit(exponential, x_data, y_data, p0=(1, 0.1))
    a, lambda_ = params
    print(f"Fitted exponential parameters: a={a}, lambda={lambda_}")

    # Plot the fitted exponential curve
    x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
    y_fit = exponential(x_fit, a, lambda_)
    ax.plot(x_fit, y_fit, color="red", label="Exponential Fit")
    ax.legend()
    ax.set_xlabel("Distance")
    ax.set_ylabel("Total Weight")
    ax.set_title(f"Total Weight vs Distance (nonzero thres {thres})")

    return fig, ax, density, lambda_


def get_ticks_and_labels(areas, duplicates):
    ticks = []
    labels = []
    for area in areas:
        if area in duplicates:
            for i in range(duplicates[area]):
                ticks.append(len(labels))
                if i == duplicates[area] // 2:
                    labels.append(f"{area}({i*2})")
                else:
                    labels.append("")
                # labels.append(f"{area}_{i+1}")
        else:
            ticks.append(len(labels))
            labels.append(area)
    return ticks, labels


def fig_3_plot_connectivity_matrix(model):
    trained_weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy()
    # mask = model.rnn.rnncell.intra_area_mask.detach().cpu().numpy()
    ticks, labels = get_ticks_and_labels(model.ce.cortical_areas, model.ce.duplicates)

    fig, ax = plt.subplots(figsize=(10, 10))

    max = np.max(np.abs(trained_weights))
    ax.imshow(
        trained_weights,
        cmap="seismic",
        vmax=max,
        vmin=-max,
    )
    ax.set_ylabel("to")
    ax.set_xlabel("from")
    ax.set_xticks(ticks, labels, rotation=90, fontsize=8, ha="center")
    ax.set_yticks(ticks, labels, fontsize=8, va="center")
    plt.tight_layout()  # Adjust layout to prevent overlap
    cbar = plt.colorbar(
        ax.imshow(trained_weights, cmap="seismic", vmax=max, vmin=-max),
        ax=ax,
        shrink=0.8,  # Adjust the shrink parameter to make the colorbar smaller
        aspect=40,  # Adjust the aspect ratio to make the colorbar thinner
    )
    return fig, ax


def calculate_densitiy(full_num_weights, nonzero_weights):
    return (1 - (full_num_weights - len(nonzero_weights)) / full_num_weights) * 100


def model_density(model):
    """
    Calculate the density of non-zero weights in the model's recurrent layer.
    """
    trained_weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy().flatten()
    full_num_weights = trained_weights.shape[0]
    thres = model.ce.zero_weights_thres
    # thres = 1e-2
    nonzero_ind, _ = remove_near_zero(trained_weights, thres)
    density = calculate_densitiy(full_num_weights, nonzero_ind)
    return density


def remove_near_zero(array, threshold):
    # array = np.array(array).T.flatten()
    indices = np.where(np.abs(array) > threshold)[0]
    values = array[indices]
    return indices, values


def calculate_spearman_exponential_fit(matrix, model, thres=0.05):
    flattened_matrix = matrix.flatten()
    full_num_weights = flattened_matrix.shape[0]
    nonzero_ind, nonzero_weights_flattened = remove_near_zero(flattened_matrix, thres)
    density = calculate_densitiy(full_num_weights, nonzero_ind)
    distance_matrix = model.ce.distance_matrix

    # Calculate Pearson correlation
    distances = distance_matrix.flatten()[nonzero_ind]
    weights = abs(nonzero_weights_flattened)
    correlation, p_value = spearmanr(distances, weights)

    # fit an exponential
    params, _ = curve_fit(exponential, distances, weights, p0=(1, 0.1))
    a, lambda_ = params

    return (correlation, p_value), (a, lambda_), density


# def fig3_weights_over_distance(model, thres, mode="show"):
#     figure = "fig3"
#     trained_weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy().flatten()
#     full_num_weights = trained_weights.flatten().shape[0]
#     nonzero_ind, nonzero_weights_flattened = remove_near_zero(trained_weights, thres)
#     density = calculate_densitiy(full_num_weights, nonzero_ind)
#     distance_matrix = model.ce.distance_matrix

#     fig, ax = plt.subplots(figsize=(10, 10))

#     ax.scatter(
#         distance_matrix.flatten()[nonzero_ind],
#         abs(nonzero_weights_flattened),
#         c="blue",
#         alpha=0.5,
#         marker="o",
#     )
#     ax.set_ylabel("Distance")
#     ax.set_ylabel("Weight")
#     ax.set_title(
#         f"non-zero weights over distance with {density:.2f}% density (thres {thres})"
#     )

#     # Calculate Pearson correlation

#     distances = distance_matrix.flatten()[nonzero_ind]
#     weights = abs(nonzero_weights_flattened)
#     correlation, p_value = pearsonr(distances, weights)

#     print(f"Pearson correlation: {correlation}, p-value: {p_value}")

#     # fit an exponential
#     params, _ = curve_fit(exponential, distances, weights, p0=(1, 0.1))
#     a, lambda_ = params
#     print(f"Fitted exponential parameters: a={a}, lambda={lambda_}")
#     x_fit = np.linspace(np.min(distances), np.max(distances), 100)
#     y_fit = exponential(x_fit, a, lambda_)
#     ax.plot(x_fit, y_fit, color="red", label="Exponential Fit")
#     ax.legend()

#     return fig, ax, correlation, p_value


def fit_spearman_and_exponential(FLN_matrix, model, thres):
    FLN_flattened = FLN_matrix.flatten()
    full_num_weights = FLN_flattened.shape[0]
    nonzero_ind, nonzero_fln_flattened = remove_near_zero(FLN_flattened, thres)
    density = calculate_densitiy(full_num_weights, nonzero_ind)
    distance_matrix = model.ce.distance_matrix / 5

    # Calculate Pearson correlation
    distances = distance_matrix.flatten()[nonzero_ind]
    FLN = abs(nonzero_fln_flattened)
    correlation, p_value = spearmanr(distances, FLN)

    # fit an exponential
    params, _ = curve_fit(exponential, distances, FLN, p0=(1, 0.1))
    a, lambda_ = params

    return (correlation, p_value), (a, lambda_)


def fig3_FLN_over_distance(FLN_matrix, model, thres, mode="show"):
    FLN_flattened = FLN_matrix.flatten()
    full_num_weights = FLN_flattened.shape[0]
    nonzero_ind, nonzero_fln_flattened = remove_near_zero(FLN_flattened, thres)
    density = calculate_densitiy(full_num_weights, nonzero_ind)
    distance_matrix = model.ce.distance_matrix / 5

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        distance_matrix.flatten()[nonzero_ind],
        abs(nonzero_fln_flattened),
        c="blue",
        alpha=0.5,
        marker="o",
    )
    ax.set_ylabel("Distance")
    ax.set_ylabel("Weight")
    ax.set_title(
        f"Weights Over Distance"
    )

    # Calculate Pearson correlation

    distances = distance_matrix.flatten()[nonzero_ind]
    FLN = abs(nonzero_fln_flattened)
    correlation, p_value = spearmanr(distances, FLN)

    print(f"Pearson correlation: {correlation}, p-value: {p_value}")

    # fit an exponential
    params, _ = curve_fit(exponential, distances, FLN, p0=(1, 0.1))
    a, lambda_ = params
    print(f"Fitted exponential parameters: a={a}, lambda={lambda_}")
    x_fit = np.linspace(np.min(distances), np.max(distances), 100)
    y_fit = exponential(x_fit, a, lambda_)
    ax.plot(x_fit, y_fit, color="red", label="Exponential Fit")
    ax.legend()

    return fig, ax, (correlation, p_value), (a, lambda_)


# Define the exponential function
def exponential(x, a, b):
    return a * np.exp(-b * x)


def exponential_fit(x, y):
    # Fit the exponential function to the data
    params, _ = curve_fit(exponential, x, y, p0=(1, 0.1))  # Initial guess for a and b
    a, lambda_ = params  # Extract the parameters
    return a, lambda_


def fig3_weight_dist_exponential_fit(model, thres=1e-2, bins=50, mode="show"):
    figure = "fig3"
    trained_weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy().flatten()
    nonzero_ind, nonzero_weights_flattened = remove_near_zero(trained_weights, thres)

    fig, ax = plt.subplots(figsize=(10, 10))

    counts, bins, ignored = ax.hist(
        abs(nonzero_weights_flattened),
        bins=bins,
        density=True,
        alpha=0.5,
        color="blue",
    )
    ax.set_ylabel("count")
    ax.set_xlabel("Weight")

    # Fit the exponential function to the data
    x_data = np.arange(len(counts))  # x values (indices of counts)
    y_data = counts  # y values (counts)

    params, _ = curve_fit(
        exponential, x_data, y_data, p0=(1, 0.1)
    )  # Initial guess for a and b
    a, lambda_ = params  # Extract the parameters

    print(f"Fitted lambda: {lambda_}")

    if mode == "save":
        os.makedirs(figure, exist_ok=True)
        plt.savefig(
            f"{figure}/weights_dist_exponential_fit.png", dpi=300, bbox_inches="tight"
        )
    if mode == "show":
        plt.show()
    if mode == "return":
        return fig, ax


def FLN_ij(weights):
    """
    Args:
        weights (np.ndarray): N x N matrix [to, from]

    Returns:
        np.ndarray
    """
    weights = np.abs(weights)
    row_sums = weights.sum(axis=1, keepdims=True)

    fln_matrix = weights / row_sums
    fln_matrix[np.isnan(fln_matrix)] = 0
    fln_matrix[np.isinf(fln_matrix)] = 0

    assert np.allclose(fln_matrix.sum(axis=1), 1, atol=1e-4)

    return fln_matrix


# def fig_3_FLN_matrix(model):
#     weigths = model.rnn.rnncell.weight_hh.detach().cpu().numpy()

#     fln_matrix = FLN_ij(weigths)

#     fig, ax = plt.subplots(figsize=(10, 10))

#     ticks, labels = get_ticks_and_labels(model.ce.cortical_areas, model.ce.duplicates)

#     ax.imshow(fln_matrix, cmap="hot", interpolation="nearest")
#     ax.set_title("FLN matrix")
#     ax.set_ylabel("to")
#     ax.set_xlabel("from")
#     ax.set_xticks(ticks, labels, rotation=90, fontsize=8, ha="center")
#     ax.set_yticks(ticks, labels, fontsize=8, va="center")
#     plt.tight_layout()
#     cbar = plt.colorbar(
#         ax.imshow(fln_matrix, cmap="hot"),
#         ax=ax,
#         shrink=0.8,
#         aspect=40,
#     )

#     return fig, ax, fln_matrix


def fig_3_FLN_matrix(model, sort=False):
    weigths = model.rnn.rnncell.weight_hh.detach().cpu().numpy()
    if sort:
        hierarchy_dict, hierarchy_boundaries, area_names_hierarchical, new_indices = (
            sort_connectivity_matrix_and_labels(model=model)
        )
        weigths = reorder_matrix(weigths, new_indices)

    fln_matrix = FLN_ij(weigths)

    # fln_matrix[np.abs(fln_matrix) < 0.05] = 0

    fln_matrix = sum_over_duplicates(fln_matrix, model.ce, mode="mean")
    model.ce.duplicates = {}

    fig, ax = plt.subplots(figsize=(10, 10))

    # ticks, labels = get_ticks_and_labels(model.ce.cortical_areas, model.ce.duplicates)
    np.fill_diagonal(fln_matrix, fln_matrix.diagonal() * 0.2)

    ax.imshow(fln_matrix, cmap="hot", interpolation="nearest", vmin=0, vmax=0.3)
    species = model.ce.species.capitalize()
    ax.set_title("FLN Connectivity Matrix", fontsize=35, fontweight="bold")
    ax.set_ylabel(f"{species} cortical areas (to)", fontsize=35)
    ax.set_xlabel(f"{species} cortical areas (from)", fontsize=35)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xticks(ticks, labels, rotation=90, fontsize=8, ha="center")
    # ax.set_yticks(ticks, labels, fontsize=8, va="center")
    plt.tight_layout()
    cbar = plt.colorbar(
        ax.imshow(
            fln_matrix,
            cmap="hot",
            vmax=0.3,
        ),
        ax=ax,
        shrink=0.8,
        aspect=40,
    )
    cbar.set_ticks([])
    # cbar.set_ticklabels([f"{cbar.vmin:.2f}", f"{cbar.vmax:.2f}"])
    cbar.ax.text(
        0.5,
        1.01,
        f"{cbar.vmax:.1f}",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=35,
        # fontweight="bold",
    )
    cbar.ax.text(
        0.5,
        -0.01,
        f"{cbar.vmin:.1f}",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=35,
        # fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_linewidth(2)

    cbar.outline.set_linewidth(2)

    return fig, ax, fln_matrix


def compress_to_180_areas(hidden_txd: np.ndarray, pl_model) -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, 180] by averaging 'duplicate' sets
    of neurons that belong to each cortical area.
    """
    ce = pl_model.model.ce  # The CorticalEmbedding instance
    cortical_areas = ce.cortical_areas
    T, n_rnn = hidden_txd.shape
    hidden_compressed = np.zeros((T, len(cortical_areas)), dtype=hidden_txd.dtype)

    for i, area in enumerate(cortical_areas):
        start_idx = ce.area2idx[area]
        dup_count = ce.duplicates.get(area, 1)
        hidden_compressed[:, i] = hidden_txd[:, start_idx : start_idx + dup_count].mean(
            axis=1
        )

    return hidden_compressed


# circle plot
def ce_idx2area(area2ix, num_units):
    """
    Convert area indices to area names.
    """
    idx2name = {ix: name for name, ix in area2ix.items()}
    for i in range(num_units):
        if i not in idx2name:
            idx2name[i] = idx2name[i - 1]
    return idx2name


def sum_over_duplicates(connectivity_matrix: np.ndarray, ce, mode="sum") -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, 180] by averaging 'duplicate' sets
    of neurons that belong to each cortical area.
    """
    cortical_areas = ce.cortical_areas
    num_units_new = len(cortical_areas)
    num_units_old = connectivity_matrix.shape[0]
    connectivity_matrix_rows_summed = np.zeros(
        (num_units_new, num_units_old), dtype=float
    )
    connectivity_matrix_summed = np.zeros((num_units_new, num_units_new), dtype=float)

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:  # if area has duplicates
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                connectivity_matrix_rows_summed[i, :] = connectivity_matrix[
                    start_idx : start_idx + dup_count, :
                ].sum(axis=0)
                if mode == "mean":
                    connectivity_matrix_rows_summed[i, :] /= dup_count

        else:
            connectivity_matrix_rows_summed[i, :] = connectivity_matrix[i, :]

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                # print(f"area {area} has {dup_count} duplicates")
                # print(connectivity_matrix_summed[:, i].shape)
                # print(
                #     connectivity_matrix[:, start_idx : start_idx + dup_count]
                #     .sum(axis=0)
                #     .shape
                # )
                connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[
                    :, start_idx : start_idx + dup_count
                ].sum(axis=1)
                if mode == "mean":
                    connectivity_matrix_summed[:, i] /= dup_count
        else:
            connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[:, i]
    return connectivity_matrix_summed


def compress_to_180_areas(hidden_txd: np.ndarray, pl_model) -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, 180] by averaging 'duplicate' sets
    of neurons that belong to each cortical area.
    """
    ce = pl_model.model.ce  # The CorticalEmbedding instance
    cortical_areas = ce.cortical_areas
    T, n_rnn = hidden_txd.shape
    hidden_compressed = np.zeros((T, len(cortical_areas)), dtype=hidden_txd.dtype)

    for i, area in enumerate(cortical_areas):
        start_idx = ce.area2idx[area]
        dup_count = ce.duplicates.get(area, 1)
        hidden_compressed[:, i] = hidden_txd[:, start_idx : start_idx + dup_count].mean(
            axis=1
        )

    return hidden_compressed


def ce_idx2area(area2ix, num_units):
    """
    Convert area indices to area names.
    """
    idx2name = {ix: name for name, ix in area2ix.items()}
    for i in range(num_units):
        if i not in idx2name:
            idx2name[i] = idx2name[i - 1]
    return idx2name


def sum_over_duplicates(connectivity_matrix: np.ndarray, ce, mode="sum") -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, 180] by averaging 'duplicate' sets
    of neurons that belong to each cortical area.
    """
    cortical_areas = ce.cortical_areas
    num_units_new = len(cortical_areas)
    num_units_old = connectivity_matrix.shape[0]
    connectivity_matrix_rows_summed = np.zeros(
        (num_units_new, num_units_old), dtype=float
    )
    connectivity_matrix_summed = np.zeros((num_units_new, num_units_new), dtype=float)

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:  # if area has duplicates
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                connectivity_matrix_rows_summed[i, :] = connectivity_matrix[
                    start_idx : start_idx + dup_count, :
                ].sum(axis=0)
                if mode == "mean":
                    connectivity_matrix_rows_summed[i, :] /= dup_count

        else:
            connectivity_matrix_rows_summed[i, :] = connectivity_matrix[i, :]

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                # print(f"area {area} has {dup_count} duplicates")
                # print(connectivity_matrix_summed[:, i].shape)
                # print(
                #     connectivity_matrix[:, start_idx : start_idx + dup_count]
                #     .sum(axis=0)
                #     .shape
                # )
                connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[
                    :, start_idx : start_idx + dup_count
                ].sum(axis=1)
                if mode == "mean":
                    connectivity_matrix_summed[:, i] /= dup_count
        else:
            connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[:, i]
    return connectivity_matrix_summed


def reorder_matrix(matrix, new_order):
    return matrix[np.ix_(new_order, new_order)]


def get_hierarchy_from_cog_overlap(cog_network_df):

    network_names = [col for col in cog_network_df.columns if col != "Row"]

    networks_dict = {key: [] for key in network_names}

    for _, row in cog_network_df.iterrows():
        network_values = row[network_names]
        max_network = network_values.idxmax()
        networks_dict[max_network].append(row["Row"][:-4])

    hierarchy = OrderedDict(
        [
            ("Visual", networks_dict["Visual"]),
            ("SomMot", networks_dict["SomMot"]),
            ("Limbic", networks_dict["Limbic"]),
            ("Salience", networks_dict["Salience"]),
            ("DorsAtt", networks_dict["DorsAtt"]),
            ("FPN", networks_dict["FPN"]),
            ("Default", networks_dict["Default"]),
        ]
    )
    return hierarchy


def get_boundaries(hierarchy):
    """
    Get the boundaries of the hierarchy.
    """
    hierarchy_boundaries = []
    count = 0
    for sublist in hierarchy.values():
        hierarchy_boundaries.append(count)
        count += len(sublist)
    return hierarchy_boundaries


def get_hierarchical_names(hierarchy):
    """
    Get the hierarchical names from the hierarchy.
    """
    area_names_hierarchical = [
        name for sublist in hierarchy.values() for name in sublist
    ]
    return area_names_hierarchical


def get_new_indices(cortical_areas, hierarchy_names_flat):
    """
    Get the new indices for the connectivity matrix based on the hierarchy.
    """
    new_indices = [hierarchy_names_flat.index(area) for area in cortical_areas]
    return new_indices


def sort_hierarchy_dict(hierarchy_dict, ce):
    distance_from_v1 = get_distance_from(
        ce.area2idx[ce.sensory[0]],
        ce.original_distance_matrix,
        ce.cortical_areas,
    )
    for key in hierarchy_dict:
        # print("area: ", key)
        hierarchy_dict[key] = sorted(
            hierarchy_dict[key],
            key=lambda area: distance_from_v1[ce.cortical_areas.index(area)],
        )
        # for area in hierarchy_dict[key]:
        #     print(area)

    return hierarchy_dict


def sort_connectivity_matrix_and_labels(model):
    """
    1. create the hierarchy from network_overlap file
    2. sort the labels by hierarchy
    3. reorder the connectivity matrix according to the sorted labels
    """
    cog_network_df = model.ce.cog_network_overlap
    # list of cortical area labels (e.g. "V1")
    cortical_areas = model.ce.cortical_areas
    # distance_from_v1 array of float of len(cortical_areas)

    hierarchy_dict = get_hierarchy_from_cog_overlap(cog_network_df)
    hierarchy_dict = sort_hierarchy_dict(hierarchy_dict, model.ce)

    hierarchy_boundaries = get_boundaries(hierarchy_dict)
    area_names_hierarchical = get_hierarchical_names(hierarchy_dict)
    new_indices = get_new_indices(cortical_areas, area_names_hierarchical)

    return hierarchy_dict, hierarchy_boundaries, area_names_hierarchical, new_indices


def compress_to_180_areas(hidden_txd: np.ndarray, pl_model) -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, 180] by averaging 'duplicate' sets
    of neurons that belong to each cortical area.
    """
    ce = pl_model.model.ce  # The CorticalEmbedding instance
    cortical_areas = ce.cortical_areas
    T, n_rnn = hidden_txd.shape
    hidden_compressed = np.zeros((T, len(cortical_areas)), dtype=hidden_txd.dtype)

    for i, area in enumerate(cortical_areas):
        start_idx = ce.area2idx[area]
        dup_count = ce.duplicates.get(area, 1)
        hidden_compressed[:, i] = hidden_txd[:, start_idx : start_idx + dup_count].mean(
            axis=1
        )

    return hidden_compressed


def ce_idx2area(area2ix, num_units):
    """
    Convert area indices to area names.
    """
    idx2name = {ix: name for name, ix in area2ix.items()}
    for i in range(num_units):
        if i not in idx2name:
            idx2name[i] = idx2name[i - 1]
    return idx2name


def sum_over_duplicates(connectivity_matrix: np.ndarray, ce, mode="sum") -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, 180] by averaging 'duplicate' sets
    of neurons that belong to each cortical area.
    """
    cortical_areas = ce.cortical_areas
    num_units_new = len(cortical_areas)
    num_units_old = connectivity_matrix.shape[0]
    connectivity_matrix_rows_summed = np.zeros(
        (num_units_new, num_units_old), dtype=float
    )
    connectivity_matrix_summed = np.zeros((num_units_new, num_units_new), dtype=float)

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:  # if area has duplicates
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                connectivity_matrix_rows_summed[i, :] = connectivity_matrix[
                    start_idx : start_idx + dup_count, :
                ].sum(axis=0)
                if mode == "mean":
                    connectivity_matrix_rows_summed[i, :] /= dup_count

        else:
            connectivity_matrix_rows_summed[i, :] = connectivity_matrix[i, :]

    for i, area in enumerate(cortical_areas):
        if area in ce.duplicates:
            if ce.duplicates[area] >= 1:
                start_idx = ce.area2idx[area]
                dup_count = ce.duplicates[area]
                # print(f"area {area} has {dup_count} duplicates")
                # print(connectivity_matrix_summed[:, i].shape)
                # print(
                #     connectivity_matrix[:, start_idx : start_idx + dup_count]
                #     .sum(axis=0)
                #     .shape
                # )
                connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[
                    :, start_idx : start_idx + dup_count
                ].sum(axis=1)
                if mode == "mean":
                    connectivity_matrix_summed[:, i] /= dup_count
        else:
            connectivity_matrix_summed[:, i] = connectivity_matrix_rows_summed[:, i]
    return connectivity_matrix_summed


def circle_plot(pretrained_model):
    # Example connectivity matrix (n_regions × n_regions)
    connectivity = abs(pretrained_model.rnn.rnncell.weight_hh.detach().cpu().numpy())
    print(connectivity)
    # Make it symmetric for this example
    np.fill_diagonal(connectivity, 0)  # Remove self-connections
    n_units = connectivity.shape[0]

    # Node labels
    ce = pretrained_model.ce
    connectivity *= 1 - ce.area_mask
    # connectivity = sum_over_duplicates(connectivity, pretrained_model.ce)

    idx2area = ce_idx2area(ce.area2idx, n_units)
    idx2area = dict(sorted(idx2area.items()))
    node_names = [val for val in idx2area.values()]

    sum_activity = True
    if sum_activity == True:
        connectivity = sum_over_duplicates(
            connectivity, pretrained_model.ce, mode="mean"
        )
        node_names = ce.cortical_areas
        node_order = node_names

    hierarchy_dict, hierarchy_boundaries, area_names_hierarchical, new_indices = (
        sort_connectivity_matrix_and_labels(model=pretrained_model)
    )

    connectivity = reorder_matrix(connectivity, new_indices)

    threshold = 0.005  # You can adjust this value based on your data
    connectivity_thresholded = connectivity.copy()
    connectivity_thresholded[connectivity_thresholded < threshold] = 0

    # node_names = ce.cortical_areas
    # node_order = sorted(idx2area.values())

    # Define the number of nodes in each group
    group_sizes = [28, 27, 14, 25, 24, 28, 34]

    # Choose a colormap and generate colors for each group
    # Define a list of RGB tuples for each group (normalized to [0,1])
    colormaps = [
        (0.8, 0.2, 0.2),  # Reds
        (0.2, 0.4, 0.8),  # Blues
        (0.2, 0.7, 0.3),  # Greens
        (1.0, 0.6, 0.2),  # Oranges
        (0.6, 0.4, 0.8),  # Purples
        (0.6, 0.6, 0.6),  # Greys
        (0.7, 0.3, 0.6),  # Magenta-like
    ]

    node_colours = []
    for size, color in zip(group_sizes, colormaps):
        node_colours.extend([color] * size)

    node_colours = []
    for size, cmap in zip(group_sizes, colormaps):
        colors = [
            cmap
        ] * size  # Use a fixed color from the colormap for all nodes in the group
        node_colours.extend(colors)

    # node_order = list(range(len(node_names)))
    # print(node_names)

    node_angles = circular_layout(
        area_names_hierarchical,
        node_order=area_names_hierarchical,
        # node_order=node_order,
        start_pos=90,
        group_boundaries=hierarchy_boundaries,
    )

    # Create a circle plot
    fig, ax = plot_connectivity_circle(
        connectivity_thresholded,
        area_names_hierarchical,
        n_lines=300,
        title="Connectivity Circle Plot",
        show=True,
        node_angles=node_angles,
        node_colors=node_colours,
        facecolor="white",
        textcolor="black",
        colormap="Reds",
        fontsize_names=5,  # Increase the font size for larger labels
    )
    # add_block_labels(
    #     ax,
    #     block_label_positions,
    #     block_labels,
    #     radius=1.2,  # Adjust the radius as needed
    #     fontsize=18,  # Adjust the font size as needed
    # )
    return fig, ax