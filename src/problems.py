"""
Définition des problèmes physiques (EDP) traités dans ce projet.

Deux équations aux dérivées partielles 1D sont étudiées sur l'ouvert (0, 1) :

1. Équation de Poisson :
       -u''(x) = f(x),  u(0) = u(1) = 0

2. Équation de Helmholtz modifiée (avec terme d'ordre 0) :
       -u''(x) + u(x) = f(x),  u(0) = u(1) = 0

Pour les deux problèmes, on choisit f(x) = sin(2*pi*x) et on connait donc la solution
"""

import torch


def f(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * torch.pi * x)


def solution_poisson(x: torch.Tensor) -> torch.Tensor:
    return f(x) / (4 * (torch.pi ** 2))


def solution_helmholtz(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * torch.pi * x) / (1 + 4 * (torch.pi ** 2))
