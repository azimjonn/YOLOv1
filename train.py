import os
import torch
import config
import model

from torch.utils.data import DataLoader
from data import CocoObjectDetection
from loss import YOLOLoss
import torchlens as tl

def main():
    device = torch.device('cuda')

    train_set = CocoObjectDetection(os.path.join(config.COCO_ROOT, 'images/train2017/'), os.path.join(config.COCO_ROOT, 'annotations/instances_train2017.json'))
    val_set = CocoObjectDetection(os.path.join(config.COCO_ROOT, 'images/val2017/'), os.path.join(config.COCO_ROOT, 'annotations/instances_val2017.json'))

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
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = yolo(imgs)
            print('std:', preds.std().item())
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            print(loss.item())

if __name__ == '__main__':
    main()