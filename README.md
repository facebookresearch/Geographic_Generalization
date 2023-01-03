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

By default, the evaluation evaluates a pretrained Resnet50 on the base set of properties (DCI) and the base set of tasks (fairness on dollarstreet, generalization on V2). These choices are encoded in configs, refer to the advanced section to learn about customization. The configs have the following structure: 

    config
    ├── base             # Hydra specifications, including experiment naming
    ├── dataset_library  # Library of all datasets compatible with this evaluation           
    ├── mode             # Hydra / Lightning specification for running locally / on clusters / testing
    ├── models           # Model specifications
    ├── property_library # Library of all properties compatible with this evaluation
    ├── property_group   # Groups / lists of properties to use in a given evaluation
    ├── task_library     # Library of all tasks compatible with this evaluation
    ├── task_group       # Groups of tasks to use in a given evaluation    
    ├── evaluate_defaults.yaml 
    └── ...


## Customization
<details>
  <summary> Changing Properties / Tasks Used </summary>

#### To change which properties are measured: 
- Option 1: Alter the list of properties in `config/property_group/base`
     ``` 
    config/property_group/base.yaml

      properties: [DCI, <property_name_here>]
    ```
  
- Option 2: create a new property group (make a new config file, ex: `config/property_group/new_property_group`, and specify it in `evaluate_defaults.yaml`
     ``` 
    config/property_group/new_property_group.yaml

      properties: [<property_name_here>]
    ```

    ``` 
    config/evaluate_defaults.yaml

      property_group: <new_property_group>
    ```

#### To change which tasks are evaluated: 
- Option 1: Alter the list of tasks in `config/task_group/base`
    ``` 
    config/task_group/base.yaml

      properties: [generalization_v2, <task_name_here>]
    ```
- Option 2: create a new task group (make a new config file, ex: `config/task_group/new_task_group`, and specify it in `evaluate_defaults.yaml`
    ``` 
    config/task_group/new_task_group.yaml

      tasks: [<task_name_here>]
    ```

    ``` 
    config/evaluate_defaults.yaml

      task_group: <new_task_group>
    ```
  
  </details>
  
  <details>
  <summary> Changing Models Used </summary>
  
#### To change which model(s) are used: 
- For non-sweep experiments, change the model in `evaluate_defaults.yaml`. You can find supported models in `config/models/`
    ``` 
    config/evaluate_defaults.yaml

      model: chosen_model
    ```
- For sweeps: change the models list in your sweep file directly, e.g. in `sh sweeps/basic_interplay_experiment.sh`
    ``` 
    sweeps/basic_interplay_experiment.yaml

      python evaluate.py -m model=resnet101,resnet18,chosen_model \
    ```   
    
</details>

## Extension
<details>
  <summary> Adding New Properties / Tasks </summary>
  
#### To add a new property: 
1) Add a config object to the property library found in `config/property_library/all.yaml` under the appropriate subsection
    ``` 
    config/property_library/all.yaml
      
      new_property_name: 
          _target_: properties.<property_type>.<class>
          logging_name: '<new_property_name>'
          dataset_names: [<dataset_name>]
    ```
2) Add the property name to the desired property_group (e.g. change 'properties' in `config/property_group/base.yaml` to include the new property)
    ``` 
    config/property_group/base.yaml

      properties: [dci, <new_property_name>]
    ```
3) Add a python class for a new property in `properties/<category>.py` (e.g. `properties/equivariance.py`), inheriting the `Property` class. The property object must return a dict[str: float] of measurements to be saved manually in a CSV.
    ``` 
    properties/<property_type>.py
        
      class NewPropertyName(Property):
        """Example Property Description"""

        def __init__(self, logging_name: str, dataset_names: list[str]):
            super().__init__(logging_name, dataset_names)

        def measure(
            self,
            config: DictConfig,
            model: ClassifierModule,
            trainer: pl.Trainer,
        ):
            #### Insert Calculation Here #### 
            # Log like this: trainer.logger.experiment.log({self.logging_name: 13})
            return {self.logging_name: 13}
    ```

#### To add a new task: 
1) Add a config object to the task library found in `config/task_library/all.yaml` under the appropriate subsection
     ``` 
    config/task_library/all.yaml
      
      new_task_name: 
          _target_: tasks.<task_type>.<class>
          logging_name: '<new_task_name>'
          dataset_names: ['<dataset_name>']
          metrics: ['AUC']
    ```
2) Add the task name to the desired task_group (e.g. change 'properties' in `config/task_group/base.yaml` to include the new task)
     ``` 
    config/task_group/base.yaml

      tasks: [generalization_v2, <new_task_name>]
    ```
3) Add a python class for a new task in `tasks/<category>.py` (e.g. `tasks/fairness.py`), inheriting the `Task` class.The task object must return a dict[str: float] of measurements to be saved manually in a CSV.

    ``` 
    tasks/<task_type>.py
        
      class NewTaskName(Task):
        def __init__(self, dataset_names: list[str], metrics: list, logging_name: str):
            super().__init__(dataset_names, metrics, logging_name)

        def evaluate(self, config: DictConfig, model: ClassifierModule, trainer: pl.Trainer):
            # Log like this:
            # trainer.logger.experiment.log(
                {self.logging_name + "_" + metric_name]}
            #)
            return

    ```
    
  </details>

<details>
    
  <summary> Adding New Models </summary>

  #### To add a new model: 
1) Add a config yaml file in `config/models/<new_model>.yaml` with a 'model_name' and a 'module' key that maps to the model target.
     ``` 
    config/models/<new_model>.yaml

        # @package _global_
        model_name: new_model_name

        module: 
          _target_: models.<model_architecture>.<file_name>.<class>
          learning_rate: 1e-4
          optimizer: adam

    ```
2) Add the model name to either `evaluate_defaults.yaml` or the sweep to include it in your run. 
    ``` 
    config/evaluate_defaults.yaml

      model: new_model_name
    ```
3) Add a python class for a new model in `models/<architecture_folder>/<new_model>.py` (e.g. `models/resnet/resnet.py`) that inherits the Classifier module. You can either keep all the models for a given architecture in one script, or separate them out into distinct files if there's more detailed implementation. Just make sure your the config target matches the path you use!
    
    ``` 
    models/<architecture_folder>/<new_model>.py
        
        from base_model import ClassifierModule
        
        class NewModelName(ClassifierModule):
            def __init__(
                self,
                timm_name: str = "",
                checkpoint_url: str = "",
            ):
                super().__init__(
                    timm_name=timm_name,
                    checkpoint_url=checkpoint_url
                )
            
            # Optional 
            def load_backbone(self):
                model = <something>
  
                return model

    ```
    
  </details>

  
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
