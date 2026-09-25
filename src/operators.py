"""
Opérateurs différentiels calculés par différentiation automatique importés avec le module pytorch
"""

import torch
from torch.func import grad


def f_forward(x: torch.Tensor, net, params, buffers) -> torch.Tensor:
    return torch.func.functional_call(net, (params, buffers), x.unsqueeze(-1)).squeeze()


# Dérivées première et seconde
f_forward_x = grad(f_forward, argnums=0)
f_forward_xx = grad(f_forward_x, argnums=0)
