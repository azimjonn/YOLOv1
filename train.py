import torch
import config
import model

from torch.utils.data import DataLoader
from data import CocoObjectDetection
from loss import yolo_loss

def main():
    device = torch.device('cuda')

    train_set = CocoObjectDetection('/data/coco/images/train2017/', '/data/coco/annotations/instances_train2017.json')
    val_set = CocoObjectDetection('/data/coco/images/val2017/', '/data/coco/annotations/instances_val2017.json')

    train_loader = DataLoader(
        train_set,
        config.BATCH_SIZE,
        num_workers=8,
        drop_last=True,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        config.BATCH_SIZE,
        num_workers=8,
        drop_last=True
    )

    yolo = model.build_net()
    yolo = yolo.to(device)

    optimizer = torch.optim.Adam(yolo.parameters(), lr=config.LEARNING_RATE)

    criterion = torch.nn.MSELoss()

    checkpoint_folder = 'checkpoints'

    for epoch in range(config.EPOCHS):
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = yolo(imgs)

            optimizer.zero_grad()
            loss = criterion(preds, targets)
            loss.backward()

            optimizer.step()

            print(loss.item())

if __name__ == '__main__':
    main()