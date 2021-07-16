import torchvision.transforms as T
import seaborn as sns
from PIL import ImageColor


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def map_labels_colors(labels):
    map_color = {}
    colors_hex = sns.color_palette("Paired", len(labels)).as_hex()
    for label, col in zip(labels, colors_hex):
        map_color[label] = ImageColor.getcolor(col, "RGB")
    return map_color
