# Interplay of Model Properties

## Installation
In a new conda environment (`conda create -n [name] python=3.9`), run

`pip install -f https://download.pytorch.org/whl/torch_stable.html -e .`

To record the current environment: `pip freeze --exclude-editable > requirements.txt`

## Structure & Goals
These evaluation experiments are structured into sub components we call measurements. All measurements are structured the same way in the code, but there are two categories we differentiate in interpretating results: **properties** and **benefits**.

Here, we use **property** to define a quantifiable measure of the model itself that is not *directly* benefitial. For example, a model's disentanglement or equivariance are properties because they are measures of the model itself, and are only meaningful through their impact on downstream tasks. 

Conversely, we use **benefits** to describe a model's downstream behavior - a benefit we're looking for the model to have. Examples here are fairness, OOD / adversarial robustness, generalization, etc. 

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
1) Add a config object to the measurement library found in `config/measurement_library/all.yaml` under the appropriate subsection. Measurement type is either 'properties' or 'benefits', as shown in the folder names. Leave the model and experiment_config values blank - they are dynamically passed in during the evaluation, but are necessary to list in the config for Hydra to identify the object.  
    ``` 
    config/measurement_library/all.yaml
      
      new_measurement_name: 
          _target_: measurements.<measurement_type>.<file_name>.<class>
          datamodule_names: [<datamodule_name>] # e.g. imagenet, v2
          model: 
          experiment_config: 
    ```
2) Add the measurement name to the desired measurement_group (e.g. change 'measurements' in `config/measurement_group/base.yaml` to include the new measurement)
    ``` 
    config/measurement_group/base.yaml

      measurements: [<new_measurement_name>]
    ```
3) Add a python class for a new measurement in `measurements.<measurement_type>.<file_name>.<class>`, inheriting the `Measurement` class. **For a commented and explained example, see the ClassificationAccuracyEvaluation class found [here](https://github.com/fairinternal/Interplay_of_Model_Properties/blob/d5720c589c4e151b1bcb8f9d45515bd298b885fc/measurements/benefits/generalization.py#L13).** Each measurement object is passed in a list of dataset names (that you will define in the measurement config, as above). This list determines which datasets the measurement accesses. The abstract measurement class constructs the datasets for you and stores them in the self.datamodules, which is dictionary mapping in the form of {datamodule_name: datamdule object}. To use the dataset in your measurement, just use this dictionary to access the desired datasets (see below, and in ClassificationAccuracyEvaluation example).  ** Logging: the measurement object must return a dict[str: float], with the key identifying the measurement, followng the convention of <datamodule_name>_<data_split>_<property_name>, all lowercase. Example: imagenet_test_accuracy**

    ``` 
    measurements.<measurement_type>.<file_name>.py
        
      class NewMeasurementName(Measurement):
          """<Describe the measurement>
            Args:
                datamodule_names (list[str]): list of dataset names required for this measurement. E.g. ['imagenet', 'dollarstreet']
                model (ClassifierModule): pytorch model to perform the measurement with
                experiment_config (DictConfig): Hydra config used primarily to instantiate a trainer. Must have key: 'trainer' to be compatible with pytorch lightning.
            Return:
                dict in the form {str: float}, where each key represents the name of the measurement, and each float is the corresponding value.
            """

        def __init__(self, datamodule_names: list[str],  model: ClassifierModule, experiment_config: DictConfig,):
            super().__init__(datamodule_names, model, experiment_config)

        def measure(self):

            # Get datamodule of interest
            datamodule_name, datamodule = next(iter(self.datamodules.items()))
            
            # Access model and trainer like this: self.model, self.trainer

            #### Insert Calculation Here #### 
            
            property_name = "example"
            return {f"{datamodule_name}_{split}_{property_name}: 13}
    ```    
    
    ***Common Pitfalls (Megan found in adding her own measurements):***
    - If you use a torchmetrics metric and define it outside of the Model's constructor (in our case, likely the test_step function), lightning will not handle moving it to GPU, and so you will have to when you define the metric.  

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
