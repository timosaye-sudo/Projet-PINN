"""Fonctions de visualisation utilisées pour reproduire les figures du rapport."""

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_solution_comparison(grid, solution_grid, prediction, title="", ax=None, save_path=None):
    """Trace la solution exacte contre la prédiction du réseau."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, solution_grid.detach() if hasattr(solution_grid, "detach") else solution_grid,
            label="solution réelle")
    ax.plot(grid, prediction.detach() if hasattr(prediction, "detach") else prediction,
            label="prédiction réseau", linestyle="--")
    ax.legend()
    ax.set_title(title)
    if created_fig and save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_loss_curve(losses, title="Évolution du logarithme de la perte", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.log(losses))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log(Loss)")
    ax.set_title(title)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def draw_heatmap(mat, row_labels, col_labels, label_r="", label_c="", title="", save_path=None):
    """Heatmap (échelle log) de la perte de validation en fonction de deux hyperparamètres."""
    fig = plt.figure(figsize=(len(col_labels) + 2, len(row_labels) + 2))
    sns.heatmap(
        mat, annot=True, fmt=".2e", cmap="viridis",
        xticklabels=col_labels, yticklabels=row_labels,
        norm=colors.LogNorm(), square=True, linewidths=0.1,
    )
    plt.xlabel(label_c)
    plt.ylabel(label_r)
    plt.title(title)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
