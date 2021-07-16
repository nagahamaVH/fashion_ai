import numpy as np
from tqdm import tqdm
import torch
import torchvision
from coco.coco_utils import get_coco_api_from_dataset
from coco.coco_eval import CocoEvaluator
from coco.get_metrics import get_metrics


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def train_step(epoch, device, model, dataloader, optimizer, loss_hist, use_tqdm=True):
    model.train()

    if use_tqdm:
        dataloader = tqdm(dataloader)

    loss_value = None
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = list(torch.as_tensor(image, dtype=torch.float32).to(device) for image in images)
        targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]
        model = model.to(device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if not use_tqdm:
            if batch_idx % np.ceil(len(dataloader) * 0.05).astype(int) == 0:
                print('Train Epoch: {} [{:.0f}%]'.format(epoch + 1, batch_idx / len(dataloader) * 100.))

    stats = {"loss": loss_value}

    return stats


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def eval_step(epoch, device, model, dataloader, use_tqdm=True):
    model.eval()

    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    if use_tqdm:
        dataloader = tqdm(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = list(torch.as_tensor(image, dtype=torch.float32).to(device) for image in images)
        model = model.to(device)
        outputs = model(images)

        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

        if not use_tqdm:
            if batch_idx % np.ceil(len(dataloader) * 0.05).astype(int) == 0:
                print('Validation Epoch: {} [{:.0f}%]'.format(epoch + 1, batch_idx / len(dataloader) * 100.))

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    metrics = get_metrics(coco_evaluator)

    return metrics
