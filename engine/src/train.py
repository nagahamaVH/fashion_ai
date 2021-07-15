import os
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from dataset import FashionDataset
from sklearn.preprocessing import LabelEncoder
from utils import get_transform
from engine import train_step, eval_step, Averager
from fasterrcnn_model import get_instance_segmentation_model


def train(model_name, num_epochs=100, batch_size=3, valid_size=0.2, seed=259,
          device=None, num_workers=-1, use_wandb=False,
          project_name="fashion_ai", entity="nagahamavh"):
    if use_wandb:
        wandb.init(project=project_name, entity=entity, name=model_name)
    if num_workers == -1:
        num_workers = os.cpu_count()

    data = pd.read_csv("./data/train.csv")
    # data = pd.read_csv("./data/train.csv", nrows=1000)
    # class_values = data["ClassId"].value_counts()
    # valid_classes = class_values[class_values > 30].index.tolist()
    # data = data.loc[data["ClassId"].isin(valid_classes), :].reset_index(drop=True)

    label_encoder = LabelEncoder()
    data['EncodedClass'] = label_encoder.fit_transform(data["ClassId"].values) + 1

    data_id = data.drop_duplicates(["ImageId"])

    if valid_size > 0:
        train_id, valid_id = train_test_split(
            data_id, test_size=valid_size, random_state=seed,
            stratify=data_id["EncodedClass"])

        train = data.loc[data["ImageId"].isin(train_id["ImageId"]), :].reset_index(drop=True)
        valid = data.loc[data["ImageId"].isin(valid_id["ImageId"]), :].reset_index(drop=True)

        valid_dataset = FashionDataset("./data/train", valid, 256, 256, transforms=get_transform(train=False))
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x)))
    else:
        train = data

    train_dataset = FashionDataset("./data/train", train, 256, 256, transforms=get_transform(train=True))
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x)))

    df_classes = data.loc[:, ["ClassId", "EncodedClass"]].drop_duplicates()

    num_classes = df_classes.shape[0] + 1
    model = get_instance_segmentation_model(num_classes)

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    loss_hist = Averager()

    for epoch in range(num_epochs):
        loss_hist.reset()
        train_metrics = train_step(
            device,
            model,
            train_data_loader,
            optimizer,
            loss_hist,
            use_tqdm=True)

        if valid_size > 0:
            with torch.no_grad():
                valid_metrics = eval_step(
                    device, model, valid_data_loader, use_tqdm=True)

        if use_wandb:
            if valid_size > 0:
                log_metrics = valid_metrics.copy()
                log_metrics["train_loss"] = train_metrics["loss"]
                wandb.log(log_metrics)
            else:
                wandb.log({"train_loss": train_metrics["loss"]})

    torch.save(model, os.path.join("pretrained_models", model_name + ".pth"))
    pd.DataFrame(df_classes).to_csv(os.path.join(
        "pretrained_models", model_name + ".csv"), index=False)

    if use_wandb:
        artifact_pth = wandb.Artifact(model_name.replace(" ", "-") + "_model", type='model')
        artifact_pth.add_file(os.path.join("pretrained_models", model_name + ".pth"))
        artifact_csv = wandb.Artifact(model_name.replace(" ", "-") + "_csv", type='dataset')
        artifact_csv.add_file(os.path.join("pretrained_models", model_name + ".csv"))
        wandb.log_artifact(artifact_pth)
        wandb.log_artifact(artifact_csv)


if __name__ == "__main__":
    train(model_name="model_1", num_epochs=20, use_wandb=True)
