import os
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import FashionDataset
# from utils import get_train_transform, get_valid_transform
from engine import train_step, eval_step, Averager

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(model_name, num_epochs=100, batch_size=3, valid_size=0.2, seed=259,
          device=None, num_workers=-1, use_wandb=False,
          project_name="fashion", entity="nagahamavh"):
    if use_wandb:
        wandb.init(project=project_name, entity=entity, name=model_name)
    if num_workers == -1:
        num_workers = os.cpu_count()

    # data = pd.read_csv("../data/train.csv")
    data = pd.read_csv("../data/train.csv", nrows=100)
    class_values = data["ClassId"].value_counts()
    valid_classes = class_values[class_values > 10].index.tolist()
    data = data.loc[data["ClassId"].isin(valid_classes), :].reset_index(drop=True)

    data_id = data.drop_duplicates(["ImageId"])

    if valid_size > 0:
        train_id, valid_id = train_test_split(
            data_id, test_size=valid_size, random_state=seed,
            stratify=data_id["ClassId"])

        train = data.loc[data["ImageId"].isin(train_id["ImageId"]), :].reset_index(drop=True)
        valid = data.loc[data["ImageId"].isin(valid_id["ImageId"]), :].reset_index(drop=True)

        valid_dataset = FashionDataset("../data/train", valid, 256, 256, transforms=get_transform(train=False))
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x)))
    else:
        train = data

    train_dataset = FashionDataset("../data/train", train, 256, 256, transforms=get_transform(train=True))
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x)))

    labels = data["ClassId"].unique()
    num_classes = len(labels) + 1

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

    # torch.save(model, os.path.join("pretrained_models", model_name + ".pth"))
    # pd.DataFrame(labels).to_csv(os.path.join(
    #     "pretrained_models", model_name + ".csv"), header=False, index=False)

    if use_wandb:
        artifact_pth = wandb.Artifact(model_name.replace(" ", "-") + "_model", type='model')
        artifact_pth.add_file(os.path.join("pretrained_models", model_name + ".pth"))
        artifact_csv = wandb.Artifact(model_name.replace(" ", "-") + "_csv", type='dataset')
        artifact_csv.add_file(os.path.join("pretrained_models", model_name + ".csv"))
        wandb.log_artifact(artifact_pth)
        wandb.log_artifact(artifact_csv)


if __name__ == "__main__":
    train(model_name="test", num_epochs=2)
