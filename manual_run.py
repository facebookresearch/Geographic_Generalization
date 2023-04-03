from datasets.imagenet_1k import ImageNet1kDataModule
from models.seer.seer import Seer320ClassifierModule
from models.resnet.resnet import ResNet18ClassifierModule, ResNet101ClassifierModule
from tqdm import tqdm
import torchmetrics
import torch
import torch.nn.functional as F
from transformers import RegNetForImageClassification
from datasets.geode import GeodeDataModule
import numpy as np


def run_manual_eval():
    with torch.no_grad():
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")

        dl = ImageNet1kDataModule(batch_size=32).test_dataloader()

        model = RegNetForImageClassification.from_pretrained(
            "facebook/regnet-y-320-seer-in1k"
        ).to(device)

        model.eval()

        accuracy_torchmetrics = torchmetrics.Accuracy().to(device)

        for batch_idx, (input, target) in tqdm(enumerate(dl)):
            target = target.to(device)
            input = input.to(device)

            output = model(input).logits

            acc1 = 100 * accuracy_torchmetrics(
                F.softmax(output.detach(), dim=-1), target
            )
            print("Acc:", acc1.item())

        print("Accuracy: ", 100 * accuracy_torchmetrics.compute())
    return


def run_manual_eval_geode():
    with torch.no_grad():
        torch.cuda.empty_cache()
        device = torch.device("cpu")

        model = ResNet101ClassifierModule().to(device)
        dl = GeodeDataModule().test_dataloader()

        model.eval()

        all_accs = []
        for batch_idx, (input, target, id) in tqdm(enumerate(dl)):

            output = model(input)
            confidences5, indices5 = torch.nn.functional.softmax(output, dim=-1).topk(5)

            for i in range(len(target)):
                y_int = [int(x) for x in target[i].split(",")]
                acc5 = len(set(y_int) & set(indices5[i].tolist())) > 0
                all_accs.append(acc5)

        print(np.mean(all_accs))

    return np.mean(all_accs)


if __name__ == "__main__":
    run_manual_eval_geode()
