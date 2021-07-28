import torchvision

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
