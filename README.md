# OCL Survey Code Base Instructions

# Installation

Create a new environment with python 3.10

```
conda create -n ocl_survey python=3.10
conda activate ocl_survey
```

Initialize and pull avalanche submodule

```
git submodule init
git submodule update
```

Then, install avalanche from the pulled repository

```
cd avalanche.git
pip install .
```

Install specific ocl_survey repo dependencies

```
cd ../
pip install -r requirements.txt
```

Set your PYTHONPATH as the root of the project

```
conda env config vars set PYTHONPATH=/home/.../ocl_survey
```

In order to let the scripts know where to fetch and log data, you should also create a deploy config, indicating results and dataset paths. Either add a new one or change the content of config/deploy/default.yaml

Lastly, test the environment by launching main.py

```
cd experiments/
python main.py strategy=er experiment=split_cifar100
```

# Structure

The code is structured as follows:

```
├── avalanche.git # Avalanche-Lib code
├── config # Hydra config files
│   ├── benchmark
│   ├── best_configs # Best configs found by main_hp_tuning.py are stored here
│   ├── deploy # Contains machine specific results and data path
│   ├── evaluation # Manage evaluation frequency and parrallelism
│   ├── experiment # Manage general experiment settings
│   ├── model
│   ├── optimizer
│   ├── scheduler
│   └── strategy
├── experiments
│   ├── main_hp_tuning.py # Main script used for hyperparameter optimization
│   ├── main.py # Main script used to launch single experiments
│   └── spaces.py
├── notebooks
├── results # Exemple results structure containing results for ER
├── scripts
    └── get_results.py # Easily collect results from multiple seeds
├── src
│   ├── factories # Contains the Benchmark, Method, and Model creation
│   ├── strategies # Contains code for additional strategies or plugins
│   └── toolkit
└── tests
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
