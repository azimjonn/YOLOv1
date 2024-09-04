import torch

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

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou

def keep_box(pred, C):
    '''
    pred: Tensor with shape (batch_size, S, S, C + 5 * B)
    C: int, number of classes

    Returns:
    Tensor shaped (batch_size, S, S, 6)
    last dimension represents (class, confidence, x, y, w, h) - coordinates and sides relative to whole image
    '''
    batch_size, S, _, depth = pred.shape
    B = (depth - C) // 5

    boxes = torch.zeros(batch_size, S, S, 6)

    