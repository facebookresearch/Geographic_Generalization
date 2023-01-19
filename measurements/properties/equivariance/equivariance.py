from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from measurements.properties.equivariance import transformations
from hydra.utils import instantiate
from models.resnet.resnet import ResNet18dClassifierModule
from datasets.dummy import DummyDataModule


class Equivariance(Measurement):
    """Measures equivariance of model embeddings with respect to augmentations.

    Assumes model has forward_features() function
    """

    def __init__(
        self,
        logging_name: str,
        dataset_names: list[str],
        transformation_name: str = "rotate",
    ):
        super().__init__(logging_name, dataset_names)
        # TODO: why do we need this logging_name as an argument?

        self.transformation_name = transformation_name
        self.model = ResNet18dClassifierModule()

        # samples x embedding_dim
        self.z = torch.empty(0)
        # samples x embedding_dim x number of transformation parameters
        self.z_t = torch.empty(0)
        self.z_t_shuffled = torch.empty(0)

    def reset_stored_z(self):
        self.z = torch.empty(0)
        self.z_t = torch.empty(0)
        self.z_t_shuffled = torch.empty(0)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        z = self.model.forward_features(x)
        print(f"{x.shape=}")
        print(f"{z.shape=}")

        z_t = None

        for magnitude_idx in range(10):
            transform = transformations.Transformation(
                self.transformation_name, magnitude_idx
            )
            x_i_t = transform(x)
            z_i_t = self.model.forward_features(x_i_t)
            print(f"{z_i_t.shape=}")
            if z_t is None:
                z_t = z_i_t.unsqueeze(-1)
            else:
                z_t = torch.cat([z_t, z_i_t.unsqueeze(-1)], dim=-1)
            print(f"{z_t.shape=}")

        self.z = torch.cat([self.z, z])
        self.z_t = torch.cat([self.z_t, z_t])
        return None

    def on_test_end(self):
        """Shuffle z_t"""
        z_t_shuffled = self.shuffle_z_t(self.z_t)
        self.z_t_shuffled = z_t_shuffled

    def shuffle_z_t(self, z_t: torch.Tensor) -> torch.Tensor:
        """Returns a shuffled version of z_t per column"""
        z_t_shuffled = torch.clone(z_t)
        for i in range(10):
            perm = torch.randperm(self.model.embedding_dim)
            z_t_shuffled[:, :, i] = z_t[:, perm, i]
        return z_t_shuffled

    def measure_equivariance(self) -> float:
        return 0.0

    def measure_invariance(self) -> float:
        return 0.0

    def measure(
        self,
        config: DictConfig,
        model_config: dict,
        limit_test_batches: float = 1.0,
    ) -> dict[str:float]:
        # TODO: make hydra instantiation work
        # self.model = instantiate(model_config)
        self.model.test_step = self.test_step

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            gpus=gpus,
            limit_test_batches=limit_test_batches,
        )

        dm = DummyDataModule()

        trainer.test(
            self.model,
            datamodule=dm,
        )
        results = {
            f"equivariance_{self.transformation_name}": self.measure_equivariance(),
            f"invariance_{self.transformation_name}": self.measure_invariance(),
        }
        return results
