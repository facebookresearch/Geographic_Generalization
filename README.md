# Interplay of Model Properties

## Installation
In a new conda environment (`conda create -n [name] python=3.9`), run

`pip install -f https://download.pytorch.org/whl/torch_stable.html -e .`

To record the current environment: `pip freeze --exclude-editable > requirements.txt`

## Structure & Goals
These evaluation experiments are structured into sub components we call measurements. All measurements are structured the same way in the code, but there are two categories we differentiate in interpretating results: **properties** and **benefits**.

Here, we use **property** to define a quantifiable measure of the model itself that is not *directly* benefitial. For example, a model's disentanglement or equivariance are properties because they are measures of the model itself, and are only meaningful through their impact on downstream tasks. 

Conversely, we use **tasks** to describe a model's downstream behavior - a benefit we're looking for the model to have. Examples here are fairness, OOD / adversarial robustness, generalization, etc. 

Broadly, the goal is to build an empirical understanding betweeen what we can measure about a vision model in isolation (properties), and the behaviors we're hoping to see when the model interacts with the world. 

## Running an Evaluation
To run an evaluation on a model: `python evaluate.py`

To run an evaluation over several models, use a sweep: `sh sweeps/basic_interplay_experiment.sh`

By default, the evaluation evaluates a pretrained Resnet50 on the base set of measurements (Imagenet V2 performance). These choices are encoded in configs, refer to the advanced section to learn about customization. The configs have the following structure: 

    config
    ├── base                # Hydra specifications, including experiment naming
    ├── dataset_library     # Library of all datasets compatible with this evaluation           
    ├── mode                # Hydra / Lightning specification for running locally / on clusters / testing
    ├── models              # Model specifications
    ├── measurement_library # Library of all measurements compatible with this evaluation
    ├── measurement_group   # Groups / lists of properties to use in a given evaluation   
    ├── evaluate_defaults.yaml 
    └── ...


## Customization
<details>
  <summary> Changing Which Measurements Are Used </summary>

#### To change which measurements are measured: 
- Option 1: Alter the list of measurements in `config/measurement_group/base`
     ``` 
    config/measurement_group/base.yaml

      measurements: [<add_measurement_name>]
    ```
  
- Option 2: create a new measurement group (make a new config file, ex: `config/measurement_group/new_measurement_group.yaml`, and specify it in `evaluate_defaults.yaml`
     ``` 
    config/measurement_group/new_measurement_group.yaml

      measurements: [<measurement_name>]
    ```

    ``` 
    config/evaluate_defaults.yaml

      property_group: <new_measurement_group>
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
  <summary> Adding New Measurements </summary>
  
#### To add a new measurement: 
1) Add a config object to the measurement library found in `config/measurement_library/all.yaml` under the appropriate subsection. Measurement type is either 'properties' or 'benefits', as shown in the folder names. 
    ``` 
    config/measurement_library/all.yaml
      
      new_measurement_name: 
          _target_: measurements.<measurement_type>.<file_name>.<class>
          logging_name: '<new_measurement_name>'
          dataset_names: [<dataset_name>]
    ```
2) Add the measurement name to the desired measurement_group (e.g. change 'measurements' in `config/measurement_group/base.yaml` to include the new measurement)
    ``` 
    config/measurement_group/base.yaml

      measurements: [<new_measurement_name>]
    ```
3) Add a python class for a new measurement in `measurements.<measurement_type>.<file_name>.<class>`, inheriting the `Measurement` class. **For a commented and explained example, see the StandardEvaluations task found in measurements/benefits/generalization.py.** Also, note that the measurement object must return a dict[str: float] of measurements to be saved manually in a CSV.

Common Pitfalls (Megan found in adding her own measurements):
- If you use a torchmetrics metric and define it outside of the Model's constructor (in our case, likely the test_step function), lightning will not handle moving it to GPU, and so you will have to when you define the metric.  


    ``` 
    measurements.<measurement_type>.<file_name>.py
        
      class NewMeasurementName(Measurement):
        """Example Measurement Description"""

        def __init__(self, logging_name: str, dataset_names: list[str]):
            super().__init__(logging_name, dataset_names)

        def measure(
            self,
            config: DictConfig,
            model_config: dict,
        ):
            #### Insert Calculation Here #### 
            
            return {self.logging_name +'_val': 13}
    ```    
  </details>

<details>
    
  <summary> Adding New Models </summary>

  #### To add a new model: 
1) Add a config yaml file in `config/models/<new_model>.yaml` with a 'model_name' and a 'model' key that maps to the model target.
     ``` 
    config/models/<new_model>.yaml

        # @package _global_
        model_name: new_model_name

        model: 
          _target_: models.<model_architecture>.<file_name>.<class>
          learning_rate: 1e-4
          optimizer: adam

    ```
2) Add the model name to either `evaluate_defaults.yaml` or the sweep to include it in your run. 
    ``` 
    config/evaluate_defaults.yaml

      model: new_model_name
    ```
3) Add a python class for a new model in `models/<architecture_folder>/<new_model>.py` (e.g. `models/resnet/resnet.py`) that inherits the ClassifierModule class. You can either keep all the models for a given architecture in one script, or separate them out into distinct files if there's more detailed implementation. Just make sure your the config target matches the path you use!
    
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
