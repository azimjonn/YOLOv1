import torch
import config

import torch.nn as nn
import torchlens as tl

def build_net():
    _depth = (config.B * 5 + config.C)

    yolo = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
        nn.LeakyReLU(negative_slope=0.1),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.MaxPool2d(kernel_size=2),

        *[
            nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ) for _ in range(4)
        ],
        nn.Conv2d(512, 512, 1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(512, 1024, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.MaxPool2d(2, 2),

        *[
            nn.Sequential(
                nn.Conv2d(1024, 512, 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ) for _ in range(2)
        ],
        nn.Conv2d(1024, 1024, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(1024, 1024, 3, 2, 1),
        nn.LeakyReLU(negative_slope=0.1),

        nn.Conv2d(1024, 1024, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(1024, 1024, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),

        nn.Flatten(),
        nn.Linear(50176, 4096),
        nn.LeakyReLU(0.1),

        nn.Linear(4096, config.S * config.S * _depth),
        nn.Unflatten(dim=1, unflattened_size=(config.S, config.S, _depth))
    )

    return yolo

if __name__ == "__main__":
    img = torch.randn((16, 3, 448, 448))
    yolo = build_net()
    model_history = tl.log_forward_pass(yolo, img, layers_to_save='all', vis_opt='rolled')
    print(model_history)