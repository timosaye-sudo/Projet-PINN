"""
Deux types de réseaux de neuronnes : 

    - SNN : réseau à une seule couche cachée (Shallow Neural Network), tel
    qu'utilisé par le théorème d'approximation universelle de Cybenko (1989).
    - DeepNN : réseau multi-couches?
"""

import torch
import torch.nn as nn

ACTIVATION_FUNCTIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}


class SNN(nn.Module):
    """Réseau à une couche cachée, activation configurable (par défaut tanh)."""

    def __init__(self, n_neurons: int, activation_name: str = "tanh"):
        super().__init__()
        self.couche = nn.Linear(1, n_neurons)
        self.activation_name = activation_name
        self.activation = ACTIVATION_FUNCTIONS[activation_name]
        self.out = nn.Linear(n_neurons, 1)

    def forward(self, x):
        x = self.activation(self.couche(x))
        x = self.out(x)
        return x


class DeepNN(nn.Module):
    """Réseau multi-couches à profondeur variable, activation configurable."""

    def __init__(self, n_neurons: int, n_couches: int, activation_name: str = "tanh"):
        super().__init__()
        self.couches = nn.ModuleList()
        self.couches.append(nn.Linear(1, n_neurons))
        self.activation = ACTIVATION_FUNCTIONS[activation_name]
        for _ in range(n_couches):
            self.couches.append(nn.Linear(n_neurons, n_neurons))
        self.couches.append(nn.Linear(n_neurons, 1))

    def forward(self, x):
        x = self.couches[0](x)
        for i in range(1, len(self.couches)):
            x = self.activation(x)
            x = self.couches[i](x)
        return x
