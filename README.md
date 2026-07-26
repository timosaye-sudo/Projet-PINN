# Résolution d'équations aux dérivées partielles par réseaux de neurones (PINN & Deep Ritz)

Projet de première année — École Nationale des Ponts et Chaussées (2026), réalisé au laboratoire **CERMICS**, sous la direction de Sofiane Ezzehi.

**Auteurs** : Paul Rivaud, Timoté Missaye, Soufiane Ait Abbou, Pierre Kairo

Rapport complet : [`report/rapport_PINN.pdf`](report/rapport_PINN.pdf)

---

## Contexte

Les équations aux dérivées partielles (EDP) sont traditionnellement résolues par des méthodes de discrétisation (différences finies, éléments finis). Ce projet explore une approche récente et *mesh-free* : approximer directement la solution d'une EDP par un réseau de neurones, entraîné à minimiser un résidu physique plutôt que sur des données étiquetées.

Deux problèmes 1D sur l'ouvert (0, 1) sont étudiés :

- **Équation de Poisson** : $-u''(x) = f(x)$, $u(0) = u(1) = 0$
- **Équation de Helmholtz modifiée** : $-u''(x) + u(x) = f(x)$, $u(0) = u(1) = 0$

Deux méthodes de résolution sont comparées :

| Méthode | Principe | Avantage | Inconvénient |
|---|---|---|---|
| **PINN** (Raissi et al., 2019) | Minimise le résidu de la formulation forte de l'EDP | Convergence rapide et précise | Nécessite de calculer $u''$ (activation deux fois dérivable obligatoire) |
| **Deep Ritz** (E & Yu, 2017) | Minimise une fonctionnelle d'énergie (formulation variationnelle/faible) | Une seule dérivation nécessaire | Convergence plus lente |

## Résultats principaux

### PINN — convergence sur l'équation de Poisson (3 neurones, tanh)

![Convergence PINN](results/pinn_convergence.png)

### Choix de la fonction d'activation

tanh et sigmoid (deux fois dérivables) convergent correctement ; **ReLU échoue structurellement**, sa dérivée seconde étant nulle presque partout, ce qui rend le résidu de l'EDP non minimisable.

![Heatmap activation](results/heatmap_activation.png)

### Impact de la profondeur du réseau

Au-delà d'une quinzaine de couches, la performance se dégrade brutalement (vanishing gradient avec tanh) : un réseau peu profond suffit largement pour ce problème 1D.

![Impact profondeur](results/depth_impact.png)

### Deep Ritz — Poisson et Helmholtz

![Deep Ritz Poisson](results/deepritz_poisson.png)
![Deep Ritz Helmholtz](results/deepritz_helmholtz.png)

### Effet de la pénalisation des conditions aux limites (Helmholtz)

Une pénalisation trop faible viole les conditions aux limites (*underfitting* des bords) ; une pénalisation trop forte fait négliger l'équation sur l'intérieur du domaine.

![Effet pénalisation](results/penalization_effect.png)

Le détail théorique (formulations variationnelles, démonstrations, théorème de Lax-Milgram) est disponible dans le [rapport complet](report/rapport_PINN.pdf).

## Structure du dépôt

```
Projet-PINN/
├── README.md
├── requirements.txt
├── LICENSE
├── report/
│   └── rapport_PINN.pdf
├── src/
│   ├── problems.py         # définition des EDP (Poisson, Helmholtz) et solutions exactes
│   ├── models.py            # architectures SNN (1 couche) et DeepNN (multi-couches)
│   ├── operators.py         # dérivées 1ère/2nde par différentiation automatique
│   ├── train_pinn.py        # entraînement PINN
│   ├── train_deepritz.py    # entraînement Deep Ritz
│   └── viz.py                # visualisations (courbes, heatmap)
├── notebooks/
│   └── demo.ipynb           # notebook de démonstration, reproduit les résultats ci-dessus
└── results/                  # figures générées
```

## Installation et exécution

```bash
git clone https://github.com/timosaye-sudo/Projet-PINN.git
cd Projet-PINN
pip install -r requirements.txt
jupyter notebook notebooks/demo.ipynb
```

## Références

1. G. Cybenko, *Approximation by superpositions of a sigmoidal function*, Mathematics of Control, Signals, and Systems, 1989.
2. W. E and B. Yu, *The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems*, Communications in Mathematics and Statistics, 2018.
3. M. Raissi, P. Perdikaris, G.E. Karniadakis, *Physics-informed neural networks*, Journal of Computational Physics, 2019.

## Licence

Ce projet est sous licence MIT — voir [LICENSE](LICENSE). C'est une licence permissive standard pour un dépôt de code éducatif/portfolio : sans elle, un dépôt public reste "tous droits réservés" par défaut (le code est visible mais personne n'a le droit de le réutiliser). La licence MIT autorise explicitement la réutilisation, la modification et la redistribution du code.
