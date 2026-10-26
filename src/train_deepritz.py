import torch
from torch.func import vmap

from .operators import derivee, sortie
from .problems import f, solution_helmholtz, solution_poisson


def train_deepritz_poisson(
    net,
    n_train: int = 1000,
    epochs: int = 10001,
    lr: float = 0.0005,
    penalite_bords: float = 500.0,
    silent: bool = True,
):
    grille = torch.linspace(0, 1, n_train)
    f_grille = vmap(f, in_dims=(0,))(grille)
    solution_grille = vmap(solution_poisson, in_dims=(0,))(grille)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    bord_0 = torch.tensor([0.0])
    bord_1 = torch.tensor([1.0])

    historique = {"epoch": [], "perte": [], "norme_l2": [], "etats": {}}

    for epoch in range(epochs):
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())

        u_x_grille = vmap(derivee, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )
        u_grille = vmap(sortie, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )

        # énergie 1/2 int u'^2 - int f u, plus la pénalisation aux bords
        perte = 0.5 * (u_x_grille**2).mean() - (u_grille * f_grille).mean()
        perte += penalite_bords * (
            sortie(bord_0, net, params, buffers) ** 2
            + sortie(bord_1, net, params, buffers) ** 2
        )

        optimizer.zero_grad()
        perte.backward()
        optimizer.step()

        if epoch % 2000 == 0:
            with torch.no_grad():
                norme_l2 = torch.sqrt(
                    torch.mean((solution_grille - u_grille) ** 2)
                ).item()
            historique["epoch"].append(epoch)
            historique["perte"].append(perte.item())
            historique["norme_l2"].append(norme_l2)
            historique["etats"][epoch] = u_grille.detach().clone()
            if not silent:
                print(
                    f"epoch {epoch}, perte : {perte.item():.4e}, norme L2 : {norme_l2:.4e}"
                )

    with torch.no_grad():
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        prediction = vmap(sortie, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )

    return {
        "grille": grille,
        "solution_grille": solution_grille,
        "prediction": prediction,
        "historique": historique,
    }


def train_deepritz_helmholtz(
    net,
    alpha: float,
    beta: float,
    n_train: int = 1000,
    epochs: int = 1000,
    lr: float = 0.01,
    silent: bool = True,
):
    grille = torch.linspace(0, 1, n_train)
    f_grille = vmap(f, in_dims=(0,))(grille)
    solution_grille = vmap(solution_helmholtz, in_dims=(0,))(grille)

    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    bord_0 = torch.tensor([0.0])
    bord_1 = torch.tensor([1.0])

    historique = {"epoch": [], "perte": [], "norme_l2": [], "etats": {}}

    for epoch in range(epochs):
        u_x_grille = vmap(derivee, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )
        u_grille = vmap(sortie, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )

        # énergie 1/2 int u'^2 + 1/2 int u^2 - int f u, plus la pénalisation aux bords
        perte = (
            0.5 * (u_x_grille**2).mean()
            + 0.5 * (u_grille**2).mean()
            - (u_grille * f_grille).mean()
            + alpha * sortie(bord_0, net, params, buffers) ** 2
            + beta * sortie(bord_1, net, params, buffers) ** 2
        )

        if epoch % 100 == 0:
            with torch.no_grad():
                norme_l2 = torch.sqrt(
                    torch.mean((solution_grille - u_grille) ** 2)
                ).item()
            historique["epoch"].append(epoch)
            historique["perte"].append(perte.item())
            historique["norme_l2"].append(norme_l2)
            historique["etats"][epoch] = u_grille.detach().clone()
            if not silent:
                print(
                    f"epoch {epoch}, perte : {perte.item():.4e}, norme L2 : {norme_l2:.4e}"
                )

        optimizer.zero_grad()
        perte.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = vmap(sortie, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )
        norme_l2_finale = torch.sqrt(
            torch.mean((solution_grille - prediction) ** 2)
        ).item()

    return {
        "grille": grille,
        "solution_grille": solution_grille,
        "prediction": prediction,
        "norme_l2_finale": norme_l2_finale,
        "historique": historique,
    }


if __name__ == "__main__":
    from .models import DeepNN

    torch.manual_seed(0)
    resultat = train_deepritz_helmholtz(DeepNN(5, 10), alpha=10, beta=10)
    print("norme L2 finale :", resultat["norme_l2_finale"])  # environ 3.9e-3
