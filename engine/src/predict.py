import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from fasterrcnn_model import get_instance_segmentation_model


def predict(image, model_name, device=None):
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

    info = {}
    with torch.no_grad():
        raw_info = model([T.ToTensor()(image).to(device)])[0]

    info["boxes"] = raw_info["boxes"].cpu().detach().numpy()
    encoded_labels = raw_info["labels"].cpu().detach().numpy()
    labels = [dict_classes.get(x, x) for x in encoded_labels]
    info["labels"] = labels
    info["scores"] = raw_info["scores"].cpu().detach().numpy()
    info["masks"] = raw_info["masks"].cpu().detach().numpy()

    return info


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = "../data/train/0a0f94b4e785bb2326dd1832303ce8de.jpg"
    # image_path = "lixo.jpeg"
    image = Image.open(image_path).convert('RGB')
    info = predict(image, "test_20epoch")

    plt.imshow(image)
    plt.show()

    print(info)

    detected_image = Image.fromarray(info['masks'][0, 0] * 255)
    print(info["masks"].shape)
    plt.imshow(detected_image)
    plt.show()
