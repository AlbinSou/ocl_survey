# OCL Survey Code Base Instructions

# Install instructions

Install avalanche for developers as indicated on https://avalanche.continualai.org/getting-started/how-to-install

Install ray and hydra

```
pip install hydra-core
pip install -U "ray[tune]"
```

# Structure

The code is structured as follows:

```
- src/ # contains source code for the experiments
    - factories/ # Contains the code to create models, strategies, and benchmarks. Most code additions should be done here
        - method_factory.py
        - model_factory.py
        - benchmark_factory.py
    - main.py # Main entry point for every experiments, no modifications should be needed in this
    - main_hp_tuning.py # Main file for the hyperparameter tuning, change here the options for hp search depending on the method
- config/ # Config directory used by hydra, 
    - config.yaml # Default config for normal experiments
    - hp_config.yaml # Default config for hp selection
    - results.yaml # The main results directory, defaults to ../results, you can change this
    - strategy/ # Contains strategy-specific config files (one per strategy)
    - optimizer/ # Contains optimizer-specific config files (one per optimizer type)
    - evaluation/ # Contains evaluation config files (no evaluation, non parallel evaluation, parallel evaluation)
    - benchmarks/ # Contains benchmark relative config (one per benchmark)
        - root.yaml # Put here the root of the data dir from which the datasets will be fetched
    - scheduler/ # Contains learning rate scheduling relative args (one per scheduler)
- toolkit/ # Contains some utils functions, parallel evaluation plugins, modified strategies (hyperparameter addition) etc...
- scripts/ # Contains shell scripts for running i.e multiple seeds, linear probing
- tests/ # Some tests for special functionalities, some more should be added maybe more related to the experiments
```

# Experiments launching

To launch an experiment, start from the default config file and change the part that needs to change

```
python main.py strategy=er_ace evaluation=parallel_eval
```

It's also possible to override more fine-grained arguments

```
python main.py strategy=er_ace evaluation=parallel_eval strategy.alpha=0.7 optimizer.lr=0.05
```

Before running the script, you can display the full config with "-c job" option

```
python main.py strategy=er_ace evaluation=parallel_eval -c job
```

Results will be saved in the directory specified in results.yaml. Under the following structure:

```
<results_dir>/<strategy_name>_<benchmark_name>/<seed>/
```

# Hyperparameter selection

Modify the strategy specific search parameters, search range etc ... inside main_hp_tuning.py then run

```
python main_hp_tuning.py strategy=er_ace
```
