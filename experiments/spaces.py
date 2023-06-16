#!/usr/bin/env python3
import copy

import numpy as np
from deepmerge import always_merger
from ray import tune

classical_search_space = {
    "optimizer": {
        "lr": tune.loguniform(1e-3, 1.0),
    },
    #"strategy": {
    #    "batch_size_mem": tune.randint(10, 256),
    #},
}

# With lr ramp

# er_space = {
#    "optimizer": {
#        "lr": tune.loguniform(1e-3, 1.0),
#    },
#    "strategy": {
#        "train_epochs": tune.randint(1, 10),
#        "lr_ramp": tune.sample_from(
#            lambda spec: float(
#                np.exp(
#                    np.random.uniform(
#                        np.log(1e-8),
#                        np.log((spec.config.optimizer.lr - 1e-5) / 1500),
#                    )
#                )
#            )
#        ),
#    },
# }

# ER

er_search_space = classical_search_space

# ER-ACE

er_ace_search_space_specific = {
    "strategy": {
        "alpha": tune.uniform(0.2, 0.8),
    }
}

er_ace_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), er_ace_search_space_specific
)

# DER

der_search_space_specific = {
    "strategy": {
        "alpha": tune.uniform(0.0, 1.0),
        "beta": tune.uniform(0.0, 1.0),
    }
}

der_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), der_search_space_specific
)
