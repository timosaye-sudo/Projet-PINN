import numpy as np
import torch
from torch.func import vmap

from .operators import derivee_seconde, sortie
from .problems import f, solution_poisson


def train_pinn(
    net, epochs: int = 1000, n: int = 1000, lr: float = 0.01, silent: bool = True
):
    pertes = np.zeros(epochs)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())

    grille = torch.linspace(0, 1, n)
    solution_grille = vmap(solution_poisson, in_dims=(0,))(grille)
    f_grille = vmap(f, in_dims=(0,))(grille)

    bord_0 = torch.tensor(0.0)
    bord_1 = torch.tensor(1.0)

    etats = {}

    for epoch in range(epochs):
        u_xx_grille = vmap(derivee_seconde, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )

        # résidu de -u'' = f, donc u'' + f doit valoir 0
        perte = ((u_xx_grille + f_grille) ** 2).mean()
        perte += sortie(bord_0, net, params, buffers) ** 2
        perte += sortie(bord_1, net, params, buffers) ** 2
        pertes[epoch] = perte.item()

        if epoch % 100 == 0:
            with torch.no_grad():
                u_grille = vmap(sortie, in_dims=(0, None, None, None))(
                    grille, net, params, buffers
                )
            etats[epoch] = u_grille.detach().clone()
            if not silent:
                print(f"epoch {epoch}, perte : {perte.item():.4e}")

        optimizer.zero_grad()
        perte.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = vmap(sortie, in_dims=(0, None, None, None))(
            grille, net, params, buffers
        )
        norme_l2 = torch.sqrt(torch.mean((solution_grille - prediction) ** 2)).item()

    return {
        "grille": grille,
        "solution_grille": solution_grille,
        "pertes": pertes,
        "etats": etats,
        "prediction": prediction,
        "norme_l2": norme_l2,
    }


def train_pinn_with_validation(
    net,
    n_train: int = 1000,
    n_val: int = 1000,
    epochs: int = 1000,
    lr: float = 0.01,
    silent: bool = True,
):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())

    grille_train = torch.linspace(0, 1, n_train)
    grille_val = torch.linspace(0, 1, n_val)
    f_train = vmap(f, in_dims=(0,))(grille_train)
    f_val = vmap(f, in_dims=(0,))(grille_val)

    bord_0 = torch.tensor(0.0)
    bord_1 = torch.tensor(1.0)

    perte_val_min = float("inf")
    meilleurs_params, meilleurs_buffers = None, None

    for epoch in range(epochs):
        u_xx_train = vmap(derivee_seconde, in_dims=(0, None, None, None))(
            grille_train, net, params, buffers
        )
        # résidu de -u'' = f, donc u'' + f doit valoir 0
        perte_train = ((u_xx_train + f_train) ** 2).mean()
        perte_train += sortie(bord_0, net, params, buffers) ** 2
        perte_train += sortie(bord_1, net, params, buffers) ** 2

        optimizer.zero_grad()
        perte_train.backward()
        optimizer.step()

        with torch.no_grad():
            u_xx_val = vmap(derivee_seconde, in_dims=(0, None, None, None))(
                grille_val, net, params, buffers
            )
            perte_val = ((u_xx_val + f_val) ** 2).mean()
            perte_val += sortie(bord_0, net, params, buffers) ** 2
            perte_val += sortie(bord_1, net, params, buffers) ** 2

            if perte_val.item() < perte_val_min:
                perte_val_min = perte_val.item()
                meilleurs_params = {k: v.clone() for k, v in params.items()}
                meilleurs_buffers = {k: v.clone() for k, v in buffers.items()}

        if epoch % 100 == 0 and not silent:
            print(
                f"epoch {epoch}, perte train : {perte_train.item():.4e}, perte val : {perte_val.item():.4e}"
            )

    params_finaux = meilleurs_params if meilleurs_params is not None else params
    buffers_finaux = meilleurs_buffers if meilleurs_buffers is not None else buffers

    with torch.no_grad():
        solution_grille = vmap(solution_poisson, in_dims=(0,))(grille_train)
        prediction = vmap(sortie, in_dims=(0, None, None, None))(
            grille_train, net, params_finaux, buffers_finaux
        )
        norme_l2 = torch.sqrt(torch.mean((solution_grille - prediction) ** 2)).item()

    return {
        "perte_val_min": perte_val_min,
        "norme_l2": norme_l2,
        "meilleurs_params": meilleurs_params,
        "meilleurs_buffers": meilleurs_buffers,
        "grille": grille_train,
        "solution_grille": solution_grille,
        "prediction": prediction,
    }


if __name__ == "__main__":
    from .models import SNN

    torch.manual_seed(0)
    resultat = train_pinn(SNN(3))
    print("norme L2 finale :", resultat["norme_l2"])  # environ 1.5e-3
