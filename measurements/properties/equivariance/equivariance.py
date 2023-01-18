from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from measurements.properties.equivariance import transformations
from hydra.utils import instantiate
from models.resnet.resnet import ResNet18dClassifierModule


class Equivariance(Measurement):
    """Measures equivariance of model embeddings with respect to augmentations.

    Assumes model has forward_features() function
    """

    def __init__(
        self,
        # TODO: why do we need this logging_name as an argument?
        logging_name: str,
        dataset_names: list[str],
        transformation_name: str = "rotate",
    ):
        super().__init__(logging_name, dataset_names)

        self.transformation_name = transformation_name
        # samples x embedding_dim
        self.z = torch.empty(0)
        # samples x embedding_dim x number of transformation parameters
        self.z_t = torch.empty(0)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        z = self.model.forward_features(x)

        z_t = torch.empty(0)

        for magnitude_idx in range(10):
            transform = transformations.Transformation(
                self.transform_name, magnitude_idx
            )
            x_i_t = transform(x)
            z_i_t = self(x_i_t)
            z_t = torch.cat([z_t, z_i_t], dim=1)

        self.z = torch.cat([self.z, z])
        self.z_t = torch.cat([self.z_t, z_t])
        return None

    def measure_equivariance(self) -> float:
        pass

    def measure_invariance(self) -> float:
        pass

    def measure(
        self,
        config: DictConfig,
        model_config: dict,
    ) -> dict[str:float]:
        # TODO: make hydra instantiation work
        # self.model = instantiate(model_config)
        self.model = ResNet18dClassifierModule()

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(gpus=gpus)

        trainer.test(self.model, self)
        results = {
            f"equivariance_{self.transformation_name}": self.measure_equivariance(),
            f"invariance_{self.transformation_name}": self.measure_invariance(),
        }
        return results
