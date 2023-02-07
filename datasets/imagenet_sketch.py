from datasets.imagenet import ImageNetDataModule


class ImageNetSketchDataModule(ImageNetDataModule):
    def __init__(
        self,
        data_dir: str = "/checkpoint/meganrichards/datasets/imagenet_sketch/",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
    ):
        """Pytorch lightning based datamodule for ImageNet Sketch dataset, found here: https://github.com/HaohanWang/ImageNet-Sketch

        Args:
            data_dir (str, optional): Path to imagenet dataset directory. Defaults to "/checkpoint/meganrichards/datasets/imagenet_sketch/".
            batch_size (int, optional): Batch size to use in datamodule. Defaults to 32.
            num_workers (int, optional): Number of workers to use in the dataloaders. Defaults to 8.
            image_size (int, optional): Side length for image crop. Defaults to 224.
        """

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
