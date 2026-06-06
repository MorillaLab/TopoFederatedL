"""
pTopoFL: Privacy-Preserving Personalised Federated Learning
via Persistent Homology
============================================================

Public API
----------
    PHDescriptor   – 48-dimensional persistent homology descriptor (§3.1)
    PTopoFL        – full pTopoFL algorithm (Algorithm 1)
    FedAvg         – McMahan et al. (2017)
    FedProx        – Li et al. (2020)
    SCAFFOLD       – Karimireddy et al. (2020)
    PFedMe         – T Dinh et al. (2020)
    IFCA           – Ghosh et al. (2020)
    make_healthcare – Scenario A data generator
    make_benchmark  – Scenario B data generator
    make_continual  – Continual FL data generator
"""

from .descriptor import PHDescriptor
from .ptopopfl   import PTopoFL
from .baselines  import FedAvg, FedProx, SCAFFOLD, PFedMe, IFCA
from .data       import make_healthcare, make_benchmark, make_continual

__version__ = "1.0.0"
__all__ = [
    "PHDescriptor",
    "PTopoFL",
    "FedAvg", "FedProx", "SCAFFOLD", "PFedMe", "IFCA",
    "make_healthcare", "make_benchmark", "make_continual",
]
