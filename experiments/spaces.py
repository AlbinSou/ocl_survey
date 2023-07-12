#!/usr/bin/env python3
import copy

import numpy as np
from deepmerge import always_merger
from ray import tune

classical_search_space = {
    "optimizer": {
        "lr": tune.loguniform(1e-5, 0.1),
    },
    #"strategy": {
    #    "train_epochs": tune.randint(3, 10),
    #}
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
        "alpha": tune.uniform(0.0, 1.0),
    }
}

er_ace_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), er_ace_search_space_specific
)

# DER

der_search_space_specific = {
    "strategy": {
        "alpha": tune.loguniform(1e-9, 10),
        "beta": tune.loguniform(1e-9, 10),
    }
}

der_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), der_search_space_specific
)

# MIR

mir_search_space_specific = {}

mir_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), mir_search_space_specific
)

# SCR

scr_search_space_specific = {
    "strategy": {
        "temperature": tune.loguniform(0.01, 2),
        #"batch_size_mem": tune.sample_from(
        #    lambda spec: np.random.randint(spec.config.strategy.train_mb_size, 500)
        #),
        #"batch_size_mem": tune.randint(10, 300),
        # "nmc_momentum": tune.loguniform(1e-9, 10),
    }
}

scr_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), scr_search_space_specific
)

# ICARL

icarl_search_space_specific = {
    "strategy": {
        "lmb": tune.loguniform(1e-9, 10),
    }
}

icarl_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), icarl_search_space_specific
)

# ER + LwF

er_lwf_search_space_specific = {
    "strategy": {
        "alpha": tune.loguniform(1e-9, 10),
        "temperature": tune.uniform(1, 2),
    }
}

er_lwf_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), er_lwf_search_space_specific
)

# RAR

rar_search_space_specific = {
    "strategy": {
        "opt_lr": tune.loguniform(1e-5, 0.1),
        "epsilon_fgsm": tune.uniform(0.01, 0.1),
        "beta_coef": tune.uniform(0.1, 1.0),
        "decay_factor_fgsm": tune.uniform(0.5, 1.0),
        "iter_fgsm": tune.randint(2, 10),
    }
}

rar_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), rar_search_space_specific
)

# AGEM

agem_search_space_specific = {
}

agem_search_space = always_merger.merge(
    copy.deepcopy(classical_search_space), agem_search_space_specific
)
