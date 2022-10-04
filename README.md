# Getting Started Template for Research

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

## Installation
`pip install -f https://download.pytorch.org/whl/torch_stable.html -e .`

To record the current environment: `pip freeze --exclude-editable > requirements.txt`

## Testing
To run tests (excluding slow tests): `python -m pytest tests/`

To run all tests (including slow tests): `python -m pytest --runslow tests/`


## Launch Experiments


`python train.py -m mode=cluster_one_gpu model=[]`

You can optionally add notes about a run `notes='exploring new learning rate'`

For other tasks, similarly run `python train_[task name].py -m +experiment=[name]`

See `config/experiment` for a list of experiments.

All logs and results can be found in `logs_dir/job_number` (defined in config).


### Relaunch a Run
To relaunch a job based on a checkpoint directory, 

`python train_[model].py -m --config-path [job_logs_dir] --config-path 'config.yaml'`

For example, `python train_video_classifier.py -m --config-path '/checkpoint/marksibrahim/logs/tmp/resnet_3d_mmnist_classifier/2021-11-02_13-12-21/0/.hydra' --config-name 'config.yaml'`

### Debugging

To check the configs for a given experiment,

`python train_[name].py -c job --resolve +experiment=[name]`

This will display the configurations used to launch the experiment. 

## Sweeps
Running hyperparamter sweeps by selecting the corresponding script from `sweeps/`. 

For exampe, `sh sweeps/sweep_resnet3d_mmnist_video_classifier.sh`

### Details
To launch experiments under your own user (instead of the team), set `wandb.entity=null` 

### Debugging Configs
To debug what configs are used: `python train_[task_name].py --cfg job`

## Analysis

To analyze experimental results, `analysis/analyze_runs.py` can fetch experimental results from weights and biases.


```python

from analysis import analyze_runs

runs = analyze_runs.Runs()
```


# Development

TODO

- [ ] add dummy data module for testing
- [ ] update tests in `tests/` make sure they pass
- [ ] add DDP local support
- [ ] add logging of image validation examples with model predictions ([example to follow](https://github.com/fairinternal/NeuralCompressionInternal/blob/7ccab7632b9ba0593b3f3adcdb84f70ba7faf4c4/projects/noisy_autoencoder/experimental/quantized_autoencoder/train.py#L24-L91))
- [ ] confirm pre-emption works
- [ ] consider adding sweeps natively to hydra
