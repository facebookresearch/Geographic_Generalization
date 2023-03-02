from ast import literal_eval
import pandas as pd
import os
import torchvision.transforms as transform_lib
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

base_path = "dataset_dollarstreet/"

###### Data Processing

t = pd.read_csv(os.path.join(base_path, "images_v2_imagenet_test.csv"), index_col=0)
t["imagenet_sysnet_id"] = t["imagenet_sysnet_id"].apply(literal_eval)


def categorize_income(x):
    if x <= 200:
        return "Q4"
    elif x > 200 and x <= 684:
        return "Q3"
    elif x > 684 and x <= 1997:
        return "Q2"
    elif x > 1997 and x <= 19671:
        return "Q1"


t["Income_Group"] = t["income"].apply(categorize_income)
assert len(t) == 4308

###### Models
model = resnet18(pretrained=True)
# model = resnet34(pretrained=True)
# model = resnet50(pretrained=True)
# model = resnet101(pretrained=True)
# model = resnet152(pretrained=True)

model.eval()

###### Evaluate

basic_normalization = transform_lib.Compose(
    [
        transform_lib.Resize(256),
        transform_lib.CenterCrop(224),
        transform_lib.ToTensor(),
        transform_lib.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]
)
imagenet_normalization = transform_lib.Compose(
    [
        transform_lib.Resize(256),
        transform_lib.CenterCrop(224),
        transform_lib.ToTensor(),
        transform_lib.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


accuracies = []
for id, row in t.iterrows():
    path = os.path.join(base_path, row["imageRelPath"])
    labels = row["imagenet_sysnet_id"]

    image = basic_normalization(Image.open(path)).unsqueeze(dim=0)
    logits = model(image)

    confidences, indices = torch.nn.functional.softmax(logits, dim=-1).topk(5)
    prediction = indices.tolist()[0]  # batch size of 1
    top_5_accuracy = len(set(prediction) & set(labels)) > 0
    accuracies.append(top_5_accuracy)

t["Top5Accuracy"] = accuracies


group_accuracies = t.groupby("Income_Group")["Top5Accuracy"].mean()

fig, ax = plt.subplots(figsize=(14, 8))
bar_plot = plt.bar(group_accuracies.index, group_accuracies.values)
plt.bar_label(bar_plot, padding=3, fmt="%.2f")
plt.title("Resnet18 Accuracy By Income Group (Imagenet Transform)")
plt.ylabel("Top 5 Accuracy")
plt.xlabel("Income Quartile")
plt.show()
fig.savefig(f"resnet18_results_imagenet_transform.PNG")
