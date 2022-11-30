# Interplay of Model Properties

## Installation
In a new conda environment (`conda create -n [name] python=3.9`), run

`pip install -f https://download.pytorch.org/whl/torch_stable.html -e .`

To record the current environment: `pip freeze --exclude-editable > requirements.txt`

## Structure & Goals
These evaluation experiments are structured into two main components / stages:
1) Measuring model **properties**
2) Evaluating the model on **tasks**

Here, we use **property** to define a quantifiable measure of the model itself that is not *directly* benefitial. For example, a model's disentanglement or equivariance are properties because they are measures of the model itself, and are only meaningful through their impact on downstream tasks. 

Conversely, we use **tasks** to describe a model's downstream behavior - a benefit we're looking for the model to have. Examples here are fairness, OOD / adversarial robustness, generalization, etc. 

Broadly, the goal is to build an empirical understanding betweeen what we can measure about a vision model in isolation, and the behaviors we're hoping to see when the model interacts with the world. 

## Running an Evaluation
To run an evaluation on a model: `python evaluate.py`

To run an evaluation over several models, use a sweep: `sh sweeps/basic_interplay_experiment.sh`

## Customization
#### To change which properties are measured: 
- Alter the list of properties in `config/property_group/base`
- OR create a new property group (make a new config file, ex: `config/property_group/base2`, and specify it in `evaluate_defaults.yaml`

#### To change which tasks are evaluated: 
- Alter the list of tasks in `config/task_group/base`
- OR create a new task group (make a new config file, ex: `config/task_group/base2`, and specify it in `evaluate_defaults.yaml`

#### To change which model(s) are used: 
- For non-sweep experiments, change the model in `evaluate_defaults.yaml`. You can find supported models in `config/models/`
- For sweeps: change the models list in your sweep file directly, e.g. in `sh sweeps/basic_interplay_experiment.sh`

## Extension
#### To add a new property: 
1) Add a config object to the property library found in `config/property_library/all.yaml` under the appropriate subsection
2) Add the property name to the desired property_group (e.g. change 'properties' in `config/property_group/base.yaml` to include the new property)
3) Add a python class for a new property in `properties/<category>.py` (e.g. `properties/equivariance.py`), inheriting the `Property` class.

#### To add a new task: 
1) Add a config object to the task library found in `config/task_library/all.yaml` under the appropriate subsection
2) Add the task name to the desired task_group (e.g. change 'properties' in `config/task_group/base.yaml` to include the new task)
3) Add a python class for a new task in `tasks/<category>.py` (e.g. `tasks/fairness.py`), inheriting the `Task` class.

#### To add a new model: 
1) Add a config yaml file in `config/models/<new_model>.yaml` with a 'model_name' and a 'module' key that maps to the model target.
2) Add the model name to either `evaluate_defaults.yaml` or the sweep to include it in your run. 
3) Add a python class for a new model in `models/<architecture_folder>/<new_model>.py` (e.g. `models/resnet/resnet.py`). You can either keep all the models for a given architecture in one script, or separate them out into distinct files if there's more detailed implementation. Just make sure your the config target matches the path you use!

## Testing
To run tests (excluding slow tests): `python -m pytest tests/`

To run all tests (including slow tests): `python -m pytest --runslow tests/`

To launch a run on a few batches locally: `python train.py -m mode=local_test`

## Debugging Configs
To debug what configs are used: `python evaluate.py --cfg job`


# Benefits

- [x] support for multi-GPU/node training
- [x] handles requeuing and checkpoint using Submit/PyTorch Lightning
- [x] configs managed using Hydra
- [x] logging using Weights and Biases
- [x] includes unit testing using PyTest
- [x] includes logging of a few sample validation examples with model predictions
- [x] snapshot of code (so you can develop while jobs runs) based on [Matt Le's Sync Code](https://fb.workplace.com/groups/airesearchinfrausers/posts/1774890499334188/?comment_id=1774892729333965&reply_comment_id=1775084782648093)



# Development

TODO
- [ ] add DDP local support
- [ ] consider adding sweeps natively to hydra configs
