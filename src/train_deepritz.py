"""
Deux problèmes sont traités :

- `train_deepritz_poisson` : équation de Poisson, J(u) = 1/2 int|u'|^2 - fu
- `train_deepritz_helmholtz` : équation de Helmholtz modifiée (lambda=1),
  J(u) = 1/2 int|u'|^2 + 1/2 int|u|^2 - fu

"""

import torch
from torch.func import vmap

from .operators import f_forward, f_forward_x
from .problems import f, solution_poisson, solution_helmholtz


def train_deepritz_poisson(
    net,
    n_train: int = 1000,
    epochs: int = 10001,
    lr: float = 0.0005,
    bc_penalty: float = 500.0,
    silent: bool = True,
):
    """
    Deep Ritz appliqué à l'équation de Poisson.

    J(u) = 1/2 * int(u'(x)^2) dx - int(f(x) u(x)) dx  (+ pénalisation aux bords)
    """
    grid = torch.linspace(0, 1, n_train)
    f_grid = vmap(f, in_dims=(0,))(grid)
    solution_grid = vmap(solution_poisson, in_dims=(0,))(grid)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    bc_0 = torch.tensor([0.0])
    bc_1 = torch.tensor([1.0])

    history = {"epoch": [], "loss": [], "l2_norm": [], "snapshots": {}}

    for epoch in range(epochs):
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())

        u_x_grid = vmap(f_forward_x, in_dims=(0, None, None, None))(grid, net, params, buffers)
        u_grid = vmap(f_forward, in_dims=(0, None, None, None))(grid, net, params, buffers)

        loss = 0.5 * (u_x_grid ** 2).mean() - (u_grid * f_grid).mean()
        loss += bc_penalty * (
            f_forward(bc_0, net, params, buffers) ** 2
            + f_forward(bc_1, net, params, buffers) ** 2
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2000 == 0:
            with torch.no_grad():
                l2_norm = torch.sqrt(torch.mean((solution_grid - u_grid) ** 2)).item()
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["l2_norm"].append(l2_norm)
            history["snapshots"][epoch] = u_grid.detach().clone()
            if not silent:
                print(f"Epoch {epoch}, Loss: {loss.item():.4e}, L2 Norm: {l2_norm:.4e}")

    with torch.no_grad():
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        u_grid_final = vmap(f_forward, in_dims=(0, None, None, None))(grid, net, params, buffers)

    return {"grid": grid, "solution_grid": solution_grid, "prediction": u_grid_final, "history": history}


def train_deepritz_helmholtz(
    net,
    alpha: float,
    beta: float,
    n_train: int = 1000,
    epochs: int = 1000,
    lr: float = 0.01,
    silent: bool = True,
):
    """
    Deep Ritz appliqué à l'équation de Helmholtz modifiée (lambda = 1).

    J(u) = 1/2 * int(u'(x)^2) dx + 1/2 * int(u(x)^2) dx - int(f(x) u(x)) dx
    (+ pénalisation aux bords, coefficients alpha et beta)

    """
    grid = torch.linspace(0, 1, n_train)
    f_grid = vmap(f, in_dims=(0,))(grid)
    solution_grid = vmap(solution_helmholtz, in_dims=(0,))(grid)

    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    bc_0 = torch.tensor([0.0])
    bc_1 = torch.tensor([1.0])

    history = {"epoch": [], "loss": [], "l2_norm": [], "snapshots": {}}

    for epoch in range(epochs):
        u_x_grid = vmap(f_forward_x, in_dims=(0, None, None, None))(grid, net, params, buffers)
        u_grid = vmap(f_forward, in_dims=(0, None, None, None))(grid, net, params, buffers)

        # Correction : le terme f*u est bien soustrait (conforme au rapport)
        loss = (
            0.5 * (u_x_grid ** 2).mean()
            + 0.5 * (u_grid ** 2).mean()
            - (u_grid * f_grid).mean()
            + alpha * f_forward(bc_0, net, params, buffers) ** 2
            + beta * f_forward(bc_1, net, params, buffers) ** 2
        )

        if epoch % 100 == 0:
            with torch.no_grad():
                l2_norm = torch.sqrt(torch.mean((solution_grid - u_grid) ** 2)).item()
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["l2_norm"].append(l2_norm)
            history["snapshots"][epoch] = u_grid.detach().clone()
            if not silent:
                print(f"Epoch {epoch}, Loss: {loss.item():.4e}, L2 Norm: {l2_norm:.4e}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_grid_final = vmap(f_forward, in_dims=(0, None, None, None))(grid, net, params, buffers)
        l2_norm_final = torch.sqrt(torch.mean((solution_grid - u_grid_final) ** 2)).item()

    return {
        "grid": grid,
        "solution_grid": solution_grid,
        "prediction": u_grid_final,
        "l2_norm_final": l2_norm_final,
        "history": history,
    }
