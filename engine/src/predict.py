import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from fasterrcnn_model import get_instance_segmentation_model
from drawing import draw_segm
from utils import map_labels_colors


def _predict(image, model_name, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_classes = pd.read_csv(os.path.join("pretrained_models", model_name + ".csv"))
    dict_classes = {}
    for i in range(df_classes.shape[0]):
        dict_classes[df_classes.loc[i, "EncodedClass"]] = df_classes.loc[i, "ClassId"]

    model = get_instance_segmentation_model(df_classes.shape[0] + 1)
    model = torch.load(os.path.join("pretrained_models", model_name + ".pth"))
    model.eval()
    model.to(device)

    info = {"boxes": [], "labels": [], "scores": [], "masks": []}
    with torch.no_grad():
        raw_info = model([T.ToTensor()(image).to(device)])[0]

    raw_info["boxes"] = raw_info["boxes"].cpu().detach().numpy()
    encoded_labels = raw_info["labels"].cpu().detach().numpy()
    labels = [dict_classes.get(x, x) for x in encoded_labels]
    raw_info["labels"] = labels
    raw_info["scores"] = raw_info["scores"].cpu().detach().numpy()
    raw_info["masks"] = raw_info["masks"].cpu().detach().numpy() * 255
    for i in range(len(raw_info["labels"])):
        if raw_info["scores"][i] > 0.7:  # threshold
            info["boxes"].append(raw_info["boxes"][i].tolist())
            info["labels"].append(raw_info["labels"][i].tolist())
            info["scores"].append(raw_info["scores"][i].tolist())
            info["masks"].append(raw_info["masks"][i].tolist())

    return info


def predict(image, model_name, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = _predict(image, model_name, device)

    classes = pd.read_csv(os.path.join("pretrained_models", model_name + ".csv"))
    mapped_color = map_labels_colors(classes["ClassId"].values)
    image_proc = draw_segm(image, info, mapped_color)

    return image_proc, info


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    image_path = "../data/train/0a0f94b4e785bb2326dd1832303ce8de.jpg"
    # image_path = "lixo.jpeg"
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256), resample=Image.BILINEAR)
    image = np.asarray(image)
    image_proc, info = predict(image, "model_batch4")

    plt.imshow(image_proc)
    plt.show()

    print(info)
