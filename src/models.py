import torch
from torch import nn

ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}


class SNN(nn.Module):
    def __init__(self, n_neurons: int, nom_activation: str = "tanh"):
        super().__init__()
        self.couche = nn.Linear(1, n_neurons)
        self.activation = ACTIVATIONS[nom_activation]
        self.out = nn.Linear(n_neurons, 1)

    def forward(self, x):
        x = self.activation(self.couche(x))
        x = self.out(x)
        return x


class DeepNN(nn.Module):
    def __init__(self, n_neurons: int, n_couches: int, nom_activation: str = "tanh"):
        super().__init__()
        self.couches = nn.ModuleList()
        self.couches.append(nn.Linear(1, n_neurons))
        self.activation = ACTIVATIONS[nom_activation]
        for _ in range(n_couches):
            self.couches.append(nn.Linear(n_neurons, n_neurons))
        self.couches.append(nn.Linear(n_neurons, 1))

    def forward(self, x):
        x = self.couches[0](x)
        for i in range(1, len(self.couches)):
            x = self.activation(x)
            x = self.couches[i](x)
        return x


if __name__ == "__main__":
    x = torch.linspace(0, 1, 5).unsqueeze(-1)
    print(SNN(3)(x).shape, DeepNN(5, 2)(x).shape)  # deux fois torch.Size([5, 1])
