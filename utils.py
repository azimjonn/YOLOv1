import torch

from torchvision.transforms.functional import to_pil_image

from PIL import ImageDraw

def intersection_over_union(box1, box2):
    '''
        box1 has shape (..., 4)
        box2 has shape (..., 4)

        return shape (..., 1)
    '''
    
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:]
    box1_0 = box1_xy - box1_wh / 2  # upper left coordinate
    box1_1 = box1_xy + box1_wh / 2  # lower right coordinate

    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:]
    box2_0 = box2_xy - box2_wh / 2
    box2_1 = box2_xy + box2_wh / 2

    cross_x0 = torch.max(box1_0[..., 0], box2_0[..., 0])
    cross_y0 = torch.max(box1_0[..., 1], box2_0[..., 1])
    cross_x1 = torch.min(box1_1[..., 0], box2_1[..., 0])
    cross_y1 = torch.min(box1_1[..., 1], box2_1[..., 1])

    intersection_area = (cross_x1 - cross_x0).clamp(min=0) * (cross_y1 - cross_y0).clamp(min=0)

    box1_area = (box1_1[..., 0] - box1_0[..., 0]) * (box1_1[..., 1] - box1_0[..., 1])
    box2_area = (box2_1[..., 0] - box2_0[..., 0]) * (box2_1[..., 1] - box2_0[..., 1])

    union_area = box1_area + box2_area - intersection_area + 1e-6

    iou = intersection_area / union_area

    return iou

def prediction_to_list_boxes(pred: torch.Tensor, C):
    '''
    pred: Tensor with shape (batch_size, S, S, C + 5 * B)
    C: int, number of classes

    Returns:
    [ [#image_0 [#box_0 class, class_confidence, box_confidence, x, y, w, h], [#box_1 ...] ], [#image_1] ... ], one element per batch
    x, y, w, h are respect to image size
    '''
    batch_size, S, _, depth = pred.shape
    B = (depth - C) // 5

    # reshaping so that boxes gets own dimension
    pred_boxes = pred[..., C:] \
                .reshape(-1, S, S, B, 5)
    pred_boxes_coord = pred_boxes[..., :4]          # (batch_size, S, S, B, 4)
    pred_boxes_confidence = pred_boxes[..., 4:5]    # (batch_size, S, S, B)

    class_confidence, best_class = pred[..., :C].max(dim=3, keepdim=True) # (batch_size, S, S, 1), (batch_size, S, S, 1)
    best_confidence, best_confidence_idx = pred_boxes_confidence.max(dim=3, keepdim=True)
    
    best_confidence = best_confidence.squeeze(dim=-1)
    best_coord = torch.gather(
        pred_boxes_coord,
        dim=3,
        index=best_confidence_idx.expand(-1, -1, -1, -1, 4)     # (batch_size, S, S, 4)
    ).squeeze()

    # convert center coordinates from respect to cell to respect to whole image
    best_coord[..., 0] += torch.arange(S)
    best_coord[..., 1] += torch.arange(S).unsqueeze(-1)
    
    best_coord[..., :2] /= S

    class_confidence = class_confidence.clamp(0, 1)
    best_confidence = best_confidence.clamp(0, 1)
    one_box_per_cell = torch.concat((best_class, class_confidence, best_confidence, best_coord), dim=-1)

    as_list = one_box_per_cell.view(batch_size, S * S, 7).tolist()

    return as_list

def nms(batch_boxes, confidence_threshold=0.8):
    valid_batch_boxes = []
    for sample_boxes in batch_boxes:
        valid_sample_boxes = []
        for box in sample_boxes:
            if box[-2] <= 0 or box[-1] <= 0 or \
               box[-3] < 0 or box[-4] < 0 or \
               box[-3] > 1 or box[-4] > 1:      # check for positive w, h. and x, y inside the image
                continue
            if box[1] * box[2] > confidence_threshold:
                valid_sample_boxes.append(box)
        
        valid_batch_boxes.append(valid_sample_boxes)
    
    return valid_batch_boxes

def tensor_to_pil(image: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for c, m, s in zip(image, mean, std):
        c.mul_(s).add_(m)
    
    image = image.clamp(0, 1)

    return to_pil_image(image)

def draw_boxes(image: torch.Tensor, boxes):
    imsize = image.shape[-1]

    pil_image = tensor_to_pil(image)
    draw = ImageDraw.Draw(pil_image)

    for box in boxes:
        x, y, w, h = box[-4:]
        x *= imsize
        y *= imsize
        w *= imsize
        h *= imsize

        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2
        box = (x0, y0, x1, y1)
        # print(box)
        draw.rectangle(box, outline='red', width=2)

    return pil_image

if __name__ == '__main__':
    import os
    import config
    from torch.utils.data import DataLoader
    from data import CocoObjectDetection
    
    train_set = CocoObjectDetection(
        os.path.join(config.COCO_ROOT, 'images/val2017/'),
        os.path.join(config.COCO_ROOT, 'annotations/instances_val2017.json'),
    )

    train_loader = DataLoader(
        train_set,
        2,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
        shuffle=False
    )

    for (imgs, targets) in train_loader:
        print('imgs', imgs.shape)
        temp = prediction_to_list_boxes(targets, config.C)
        for img, boxes in zip(imgs, temp):
            boxes = filter(lambda x: x[1] > 0.5, boxes)
            res = draw_boxes(img, boxes)

            res.show()
            input()