# Does Progress On Object Recognition Benchmarks Improve Real-World Generalization?

![](/images/Figure1v4.png)

This repository supports the research conducted in the paper: Does Progress on Object Recognition Benchmarks Improve Real-World Generalization? The project explores whether improvements in object recognition models on standard benchmarks (e.g., ImageNet) translate to enhanced performance in more diverse, real-world environments. The research aims to address the challenge of geographic generalization, where models are evaluated on datasets that capture various cultural, environmental, and geographical conditions. This repository also provides tools for evaluating models on the DollarStreet and GeoDE benchmarks, which include data from a wider range of geographical locations and real-world scenarios. These benchmarks are designed to test how well models generalize to new and less controlled environments beyond traditional datasets. We welcome feedback, questions, and contributions to improve the tools and resources provided here!

## In this Repository

- #### :white_check_mark: Download 6 benchmarks with just one line of code  
- #### :white_check_mark: Use our implementations of ~100 models, from ResNet to CLIP and DinoV2.
- #### :white_check_mark: Evaluate your model with a simple, scalable script
- #### :white_check_mark: Built-in logging with TensorBoard/Lightning, & compatible to run locally and multi-GPU      

## License
This repository is Attribution-NonCommercial 4.0 International licensed, as found in the LICENSE file.

## Getting Started
1. **Clone this repository:**

    ```
    git clone https://github.com/facebookresearch/Geographic_Generalization.git
    ```

2. **Install Conda**
   Follow the Conda installation guide if Conda is not already installed on your system. (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

3. **Create and activate a new conda environment:**
    ```
    conda create -n geo_general python=3.8
    conda activate geo_general
    ```

4. **Install PyTorch:**
    ```
    pip install -f https://download.pytorch.org/whl/torch_stable.html -e .
    ```

5. **Common issues:**
   If you face issues with CUDA or GPU drivers, please follow the PyTorch troubleshooting guide. (https://pytorch.org/serve/Troubleshooting.html)

## Troubleshooting
**Example**
Problem: CUDA out-of-memory errors
**Solution:**  
Try lowering the batch size in the configuration file (`config/evaluate_defaults.yaml`):
```yaml
batch_size: 16

     

## Adding Your Own Model Weights
<details>  
    <summary> Details </summary>
    
1. Find the model's yaml file (config/model/<architecture>.yaml Example: "config/model/resnet50.yaml").
2. Make a copy of the yaml file with a unique name
3. Add the checkpoint_path parameter to specify the paths to your new weights:
    ```
    ## config/model/resnet50_myweights.yaml

    model_name: resnet50_myweights
    
    model: 
      _target_: models.resnet.resnet.ResNet50ClassifierModule
       checkpoint_path: <INSERT YOUR PATH HERE>                    <- add the relative path to your model weights

     ```
4. To run an evaluation, change the model specified in config/evalaute_defaults.yaml to your new model's name, and run 'python evaluate.py'. 
    ```
    ## config/evaluate_defaults.yaml
    
    defaults:
      - base: base
      - mode: local
      - dataset_library: all
      - model: resnet50_my_weights   <- add your new model's name
      - measurement_library: all
      - measurement_group: test

     ```
</details>

## Adding A New Architecture 
<details>
  <summary>  Details </summary>

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
            def load_model(self):
                model = <something>
  
                return model

    ```
    
  </details>

## Building on Our Codebase - Guide to Further Customization 

<details>
By default, the evaluation evaluates a pretrained Resnet50 on imagenet's validation set. These choices are encoded in config files, refer to the advanced section to learn about customization. The configs have the following structure: 

    config
    ├── base                # Hydra specifications, including experiment naming
    ├── dataset_library     # Library of all datasets compatible with this evaluation           
    ├── mode                # Hydra / Lightning specification for running locally / on clusters / testing
    ├── models              # Model specifications
    ├── measurement_library # Library of all measurements compatible with this evaluation
    ├── measurement_group   # Groups / lists of properties to use in a given evaluation   
    ├── evaluate_defaults.yaml 
    └── ...
    
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
3) Add a python class for a new measurement in `measurements.<file_name>.<class>`, inheriting the `Measurement` class. **For a commented and explained example, see the ClassificationAccuracyEvaluation class.** Each measurement object is passed in a list of dataset names (that you will define in the measurement config, as above). This list determines which datasets the measurement accesses. The abstract measurement class constructs the datasets for you and stores them in the self.datamodules, which is dictionary mapping in the form of {datamodule_name: datamdule object}. To use the dataset in your measurement, just use this dictionary to access the desired datasets (see below, and in ClassificationAccuracyEvaluation example).  ** Logging: the measurement object must return a dict[str: float], with the key identifying the measurement, followng the convention of <datamodule_name>_<data_split>_<property_name>, all lowercase. Example: imagenet_test_accuracy**

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
    
 </details>
</details>
  
## Testing
To run tests (excluding slow tests): `python -m pytest tests/`

To run all tests (including slow tests): `python -m pytest --runslow tests/`

To launch a run on a few batches locally: `python train.py -m mode=local_test`

## Debugging Configs
To debug what configs are used: `python evaluate.py --cfg job`

## Additional Resources and Documentation

### Further Reading
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [PyTorch Lightning Guide](https://pytorch-lightning.readthedocs.io/en/stable/)
- [Original Research Paper](https://arxiv.org/abs/2307.13136)
- Overview of Geographic Generalization Concepts: *(Add relevant article here)*



## Citation 
 ``` 
@article{richards2023does,
  title={Does Progress On Object Recognition Benchmarks Improve Real-World Generalization?},
  author={Richards, Megan and Kirichenko, Polina and Bouchacourt, Diane and Ibrahim, Mark},
  journal={arXiv preprint arXiv:2307.13136},
  year={2023}
}
 ``` 
