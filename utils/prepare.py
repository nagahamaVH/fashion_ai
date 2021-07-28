import json
import pandas as pd


# Generating table images as csv
train = pd.read_csv("./data/train.csv")
images_table = train.drop(["EncodedPixels", "AttributesIds", "ClassId"], axis=1)
images_table["Group"] = "train"
images_table = images_table.drop_duplicates().reset_index(drop=True)
images_table["Id"] = images_table.index.values
images_table = images_table.loc[:, ["Id", "ImageId", "Height", "Width", "Group"]]

# Generating table attributes and categories as csv
with open('./data/label_descriptions.json', 'r') as file:
    label_desc = json.load(file)

# Generating table segmentation as csv
segm_table = train.loc[:, ["ImageId", "EncodedPixels", "ClassId", "AttributesIds"]]
segm_table = segm_table.merge(images_table[["Id", "ImageId"]], on="ImageId")
segm_table = segm_table.loc[:, ["Id", "EncodedPixels", "ClassId", "AttributesIds"]]

attributes_table = pd.DataFrame(label_desc["attributes"])

categories_table = pd.DataFrame(label_desc["categories"])
categories_table["detail"] = 0
categories_table.loc[categories_table["supercategory"].isin(
    ["garment parts", "closures", "decorations"]), "detail"] = 1

images_table.to_csv("./data/images_table.csv", index=None)
attributes_table.to_csv("./data/attributes_table.csv", index=None)
categories_table.to_csv("./data/categories_table.csv", index=None)
segm_table.to_csv("./data/segmentation_table.csv", index=None)
