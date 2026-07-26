"""
Opérateurs différentiels calculés par différentiation automatique.

On utilise `torch.func` (grad + vmap) pour calculer la dérivée première et
seconde de la sortie du réseau par rapport à son entrée x, sans avoir à les
dériver analytiquement à la main. C'est le cœur de l'approche PINN.
"""

import torch
from torch.func import grad


def f_forward(x: torch.Tensor, net, params, buffers) -> torch.Tensor:
    """Appelle le réseau de manière fonctionnelle (nécessaire pour torch.func)."""
    return torch.func.functional_call(net, (params, buffers), x.unsqueeze(-1)).squeeze()


# Dérivée première et seconde de la sortie du réseau par rapport à x
f_forward_x = grad(f_forward, argnums=0)
f_forward_xx = grad(f_forward_x, argnums=0)
