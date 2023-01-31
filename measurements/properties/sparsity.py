from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from measurements.properties.equivariance import transformations
from hydra.utils import instantiate
from models.resnet.resnet import ResNet18dClassifierModule
from datasets.dummy import DummyDataModule
from torch import Tensor


class Sparsity(Measurement):
    """Measures sparsity of model embeddings as the number of fired neurons per sample.

    Assumes model has forward_features() function
    """

    def __init__(
        self,
        logging_name: str,
        dataset_names: list[str],
    ):
        super().__init__(logging_name, dataset_names)

        # model is part of the property? Looks weird, shouldn't we pass it as argument?
        self.model = ResNet18dClassifierModule() 
        self.model.test_step = self.test_step
        self.model.on_test_end = self.on_test_end
        self.z = torch.empty(0)
    
    def reset_stored_z(self):
        self.z = torch.empty(0)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        # if we make forward features all layers, we could play with the layer we compute metrics on 
        z = self.model.forward_features(x)
        self.z = torch.cat([self.z, z])
        return None

    @staticmethod
    def measure_sparsity(z: Tensor) -> float:
        """
        Args:
            z: (num samples, embedding dim)
        """


    def measure(
        self,
        config: DictConfig,
        model_config: dict,
        limit_test_batches: float = 1.0,
    ) -> dict[str:float]:

        self.reset_stored_z()

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            gpus=gpus,
            limit_test_batches=limit_test_batches,
        )

        dm = DummyDataModule()
        # do we list test datasets here ? Mark code departs from ReadMe
        # datamodule_config or model_config?
        trainer.test(
            self.model,
            datamodule=dm,
        )

        results = {
            f"sparsity": self.measure_sparsity(
                self.z
            ),
        }
        return results