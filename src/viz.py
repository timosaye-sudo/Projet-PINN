import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors


def trace_solution(
    grille, solution_grille, prediction, titre="", ax=None, fichier=None
):
    nouvelle_figure = ax is None
    if nouvelle_figure:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        grille,
        solution_grille.detach()
        if hasattr(solution_grille, "detach")
        else solution_grille,
        label="solution réelle",
    )
    ax.plot(
        grille,
        prediction.detach() if hasattr(prediction, "detach") else prediction,
        label="prédiction réseau",
        linestyle="--",
    )
    ax.legend()
    ax.set_title(titre)
    if nouvelle_figure and fichier:
        fig.tight_layout()
        fig.savefig(fichier, dpi=150)
        plt.close(fig)


def trace_pertes(pertes, titre="Évolution du logarithme de la perte", fichier=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.log(pertes))
    ax.set_xlabel("epoch")
    ax.set_ylabel("log(perte)")
    ax.set_title(titre)
    if fichier:
        fig.tight_layout()
        fig.savefig(fichier, dpi=150)
        plt.close(fig)


def trace_heatmap(
    mat,
    noms_lignes,
    noms_colonnes,
    titre_lignes="",
    titre_colonnes="",
    titre="",
    fichier=None,
):
    fig = plt.figure(figsize=(len(noms_colonnes) + 2, len(noms_lignes) + 2))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2e",
        cmap="viridis",
        xticklabels=noms_colonnes,
        yticklabels=noms_lignes,
        norm=colors.LogNorm(),
        square=True,
        linewidths=0.1,
    )
    plt.xlabel(titre_colonnes)
    plt.ylabel(titre_lignes)
    plt.title(titre)
    if fichier:
        fig.tight_layout()
        fig.savefig(fichier, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    x = np.linspace(0, 1, 100)
    trace_solution(
        x, np.sin(np.pi * x), np.sin(np.pi * x) + 0.01, titre="test de trace_solution"
    )
    plt.show()
