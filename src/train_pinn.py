"""
Entraînement de réseaux PINN (Physics-Informed Neural Networks) pour
l'équation de Poisson -u''(x) = f(x), u(0) = u(1) = 0.

La fonction de perte intègre directement le résidu de l'EDP (formulation
forte) et une pénalisation des conditions aux limites :

    E(u) = mean[(u''(x_i) + f(x_i))^2] + u(0)^2 + u(1)^2

NOTE DE CORRECTION (par rapport au notebook original) :
Le code initial calculait le résidu comme (u''(x_i) - f(x_i))^2, ce qui
imposait en réalité u''=f au lieu de -u''=f. Cette erreur était "masquée"
car la fonction `solution_poisson` utilisée pour comparer contenait elle
aussi un signe inversé, cohérent avec cette équation erronée — d'où une
convergence visuellement satisfaisante mais vers la mauvaise équation.
Les deux signes ont été corrigés conjointement (voir `problems.py`) pour
être conformes à l'équation annoncée dans le rapport (-u''=f) et à sa
solution exacte donnée en section 4.1 : u*(x) = sin(2*pi*x)/(4*pi^2).
"""

import numpy as np
import torch
from torch.func import vmap

from .operators import f_forward, f_forward_x, f_forward_xx
from .problems import f, solution_poisson


def train_pinn(net, epochs: int = 1000, n: int = 1000, lr: float = 0.01, silent: bool = True):
    """
    Entraîne un réseau PINN de base (sans jeu de validation) sur l'équation
    de Poisson. Correspond à la section 4.1 du rapport (premier test PINN).

    Retourne l'historique des pertes et la norme L2 finale par rapport à la
    solution exacte.
    """
    losses = np.zeros(epochs)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())

    grid = torch.linspace(0, 1, n)
    solution_grid = vmap(solution_poisson, in_dims=(0,))(grid)
    f_grid = vmap(f, in_dims=(0,))(grid)

    bc_0 = torch.tensor(0.0)
    bc_1 = torch.tensor(1.0)

    snapshots = {}

    for epoch in range(epochs):
        f_xx_grid = vmap(f_forward_xx, in_dims=(0, None, None, None))(grid, net, params, buffers)

        # Résidu de -u''(x) = f(x), soit u''(x) + f(x) = 0
        loss = ((f_xx_grid + f_grid) ** 2).mean()
        loss += f_forward(bc_0, net, params, buffers) ** 2
        loss += f_forward(bc_1, net, params, buffers) ** 2
        losses[epoch] = loss.item()

        if epoch % 100 == 0:
            with torch.no_grad():
                current = vmap(f_forward, in_dims=(0, None, None, None))(grid, net, params, buffers)
            snapshots[epoch] = current.detach().clone()
            if not silent:
                print(f"Epoch {epoch}, Loss: {loss.item():.4e}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = vmap(f_forward, in_dims=(0, None, None, None))(grid, net, params, buffers)
        l2_norm = torch.sqrt(torch.mean((solution_grid - final) ** 2)).item()

    return {
        "grid": grid,
        "solution_grid": solution_grid,
        "losses": losses,
        "snapshots": snapshots,
        "final_prediction": final,
        "l2_norm": l2_norm,
    }


def train_pinn_with_validation(
    net,
    n_train: int = 1000,
    n_val: int = 1000,
    epochs: int = 1000,
    lr: float = 0.01,
    silent: bool = True,
):
    """
    Entraîne un réseau PINN avec un jeu de validation, et conserve les
    meilleurs paramètres rencontrés (early-stopping implicite).
    Correspond aux sections 4.2 et 4.3 du rapport (étude des
    hyperparamètres et de la profondeur du réseau).
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())

    grid_train = torch.linspace(0, 1, n_train)
    grid_val = torch.linspace(0, 1, n_val)
    f_train = vmap(f, in_dims=(0,))(grid_train)
    f_val = vmap(f, in_dims=(0,))(grid_val)

    bc_0 = torch.tensor(0.0)
    bc_1 = torch.tensor(1.0)

    min_val_loss = float("inf")
    best_params, best_buffers = None, None

    for epoch in range(epochs):
        f_xx_train = vmap(f_forward_xx, in_dims=(0, None, None, None))(grid_train, net, params, buffers)
        # Résidu de -u''(x) = f(x), soit u''(x) + f(x) = 0
        train_loss = ((f_xx_train + f_train) ** 2).mean()
        train_loss += f_forward(bc_0, net, params, buffers) ** 2
        train_loss += f_forward(bc_1, net, params, buffers) ** 2

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            f_xx_val = vmap(f_forward_xx, in_dims=(0, None, None, None))(grid_val, net, params, buffers)
            val_loss = ((f_xx_val + f_val) ** 2).mean()
            val_loss += f_forward(bc_0, net, params, buffers) ** 2
            val_loss += f_forward(bc_1, net, params, buffers) ** 2

            if val_loss.item() < min_val_loss:
                min_val_loss = val_loss.item()
                best_params = {k: v.clone() for k, v in params.items()}
                best_buffers = {k: v.clone() for k, v in buffers.items()}

        if epoch % 100 == 0 and not silent:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4e}, Val Loss: {val_loss.item():.4e}")

    plot_params = best_params if best_params is not None else params
    plot_buffers = best_buffers if best_buffers is not None else buffers

    with torch.no_grad():
        solution_grid = vmap(solution_poisson, in_dims=(0,))(grid_train)
        prediction = vmap(f_forward, in_dims=(0, None, None, None))(grid_train, net, plot_params, plot_buffers)
        l2_norm = torch.sqrt(torch.mean((solution_grid - prediction) ** 2)).item()

    return {
        "min_val_loss": min_val_loss,
        "l2_norm": l2_norm,
        "best_params": best_params,
        "best_buffers": best_buffers,
        "grid": grid_train,
        "solution_grid": solution_grid,
        "prediction": prediction,
    }
