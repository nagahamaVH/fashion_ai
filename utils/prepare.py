import json
import pandas as pd


# Generating table images as csv
train = pd.read_csv("./data/train.csv")
images_table = train.drop(["EncodedPixels", "AttributesIds"], axis=1)
images_table["Group"] = "train"

# Generating table attributes and categories as csv
with open('./data/label_descriptions.json', 'r') as file:
    label_desc = json.load(file)

# Generating table segmentation as csv
segm_table = train.loc[:, ["ImageId",
                           "EncodedPixels", "ClassId", "AttributesIds"]]

attributes_table = pd.DataFrame(label_desc["attributes"])
categories_table = pd.DataFrame(label_desc["categories"])

images_table.to_csv("./data/images_table.csv", index=None)
attributes_table.to_csv("./data/attributes_table.csv", index=None)
categories_table.to_csv("./data/categories_table.csv", index=None)
segm_table.to_csv("./data/segmentation_table.csv", index=None)
