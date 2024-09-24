import os
import torch
import config
import model
import json

from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer

from data import CocoObjectDetection
from loss import YOLOLoss
from utils import prediction_to_list_boxes, nms, draw_boxes
from tqdm import tqdm

import random

def train_epoch(model: Module, data_loader: DataLoader, criterion: YOLOLoss, optimizer: Optimizer, epoch: int):
    model.train()
    device = next(model.parameters()).device
    loop = tqdm(data_loader, f"Epoch {epoch}")
    for imgs, targets in loop:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loop.set_postfix(loss=loss.item())

        # if random.random() < 0.05:
        #     break

def validate(model: torch.nn.Module, data_loader: DataLoader):
    model.eval()
    device = next(model.parameters()).device
    results = []
    loop = tqdm(data_loader, "Validation")
    for imgs, (ids, (heights, widths)) in loop: # batch inference
        imgs = imgs.to(device)

        with torch.no_grad():
            preds = model(imgs).cpu()
        pred_boxes = nms(prediction_to_list_boxes(preds, config.C), confidence_threshold=0.7)

        for id, height, width, boxes in zip(ids, heights, widths, pred_boxes): # per sample
            for box in boxes:
                box[3] *= width.item()
                box[4] *= height.item()
                box[5] *= width.item()
                box[6] *= height.item()

                box[3] = max(box[3] - box[5] / 2, 0)
                box[4] = max(box[4] -  box[6] / 2, 0)
                detection = {
                    "image_id": id.item(),
                    "category_id": int(box[0]),
                    "bbox": box[3:],
                    "score": box[1] * box[2]
                }

                results.append(detection)

    with open('result.json', 'w') as file:
        json.dump(results, file, indent=4)

def main():
    device = torch.device('cuda')

    train_set = CocoObjectDetection(os.path.join(config.COCO_ROOT, 'images/train2017/'), os.path.join(config.COCO_ROOT, 'annotations/instances_train2017.json'))
    val_set = CocoObjectDetection(os.path.join(config.COCO_ROOT, 'images/val2017/'), os.path.join(config.COCO_ROOT, 'annotations/instances_val2017.json'), is_test=True)

    train_loader = DataLoader(
        train_set,
        config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True
    )

    yolo = model.build_net()
    yolo = yolo.to(device)

    optimizer = torch.optim.Adam(yolo.parameters(), lr=config.LEARNING_RATE)
    criterion = YOLOLoss(S=config.S, B=config.B, C=config.C, lambda_coord=config.LAMBDA_COORD, lambda_noobj=config.LAMBDA_NOOBJ)
    checkpoint_folder = 'checkpoints'

    for epoch in range(config.EPOCHS):
        train_epoch(yolo, train_loader, criterion, optimizer, epoch)
        validate(yolo, val_loader)

if __name__ == '__main__':
    main()