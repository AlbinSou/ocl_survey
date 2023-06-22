# OCL Survey Code Base Instructions

# Install instructions (Witchery)

Create a new environment with python 3.10

```
conda create -n ocl_survey python=3.10
conda activate ocl_survey
```

- To not lose sanity:

```
conda install mamba
```

- Then, follow the steps in this order

```
mamba install matplotlib
pip install torch torchvision
mamba env update -f environment.yaml
```

- Add to your python path the ocl_survey directory and the avalanche directory

```
conda env config vars set PYTHONPATH=/home/.../ocl_survey:/home/ocl_survey/avalanche.git
```

- Add a deploy config in the config/deploy folder, precising results and dataset path
- Test the environment by launching main.py

```
cd experiments/
python main.py
```

# Structure

The code is structured as follows:

```
- src/ # contains source code for the experiments
    - factories/ # Contains the code to create models, strategies, and benchmarks. Most code additions should be done here
        - method_factory.py
        - model_factory.py
        - benchmark_factory.py
    - toolkit/ # Contains some utils functions, parallel evaluation plugins, modified strategies (hyperparameter addition) etc...
- config/ # Config directory used by hydra, 
    - config.yaml # Default config for normal experiments
    - hp_config.yaml # Default config for hp selection
    - results.yaml # The main results directory, defaults to ../results, you can change this
    - strategy/ # Contains strategy-specific config files (one per strategy)
    - optimizer/ # Contains optimizer-specific config files (one per optimizer type)
    - evaluation/ # Contains evaluation config files (no evaluation, non parallel evaluation, parallel evaluation)
    - benchmarks/ # Contains benchmark relative config (one per benchmark)
    - experiments/ # Contains the overrides for given experiments (one per benchmark), i.e model, batch size etc..
    - deploy/ # Folder to precise results and dataset path
    - scheduler/ # Contains learning rate scheduling relative args (one per scheduler)
- experiments/
    - main.py # Main entry point for every experiments, no modifications should be needed in this
    - main_hp_tuning.py # Main file for the hyperparameter tuning, change here the options for hp search depending on the method
- scripts/ # Contains shell scripts for running i.e multiple seeds, linear probing
- tests/ # Some tests for special functionalities, some more should be added maybe more related to the experiments
```

# Experiments launching

To launch an experiment, start from the default config file and change the part that needs to change

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel
```

It's also possible to override more fine-grained arguments

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel strategy.alpha=0.7 optimizer.lr=0.05
```

Before running the script, you can display the full config with "-c job" option

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel -c job
```

Results will be saved in the directory specified in results.yaml. Under the following structure:

```
<results_dir>/<strategy_name>_<benchmark_name>/<seed>/
```

# Hyperparameter selection

Modify the strategy specific search parameters, search range etc ... inside main_hp_tuning.py then run

```
python main_hp_tuning.py strategy=er_ace experiment=split_cifar100
```
