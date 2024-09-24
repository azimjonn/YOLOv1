import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=80, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, predictions, targets: torch.Tensor):
        target_class = targets[..., :self.C]                                # (batch_size, S, S, C)
        target_box_coord = targets[..., self.C:self.C+4]                    # (batch_size, S, S, 4)
        target_confidence = targets[..., self.C+4:self.C+5]                 # (batch_size, S, S, 1)

        pred_class = predictions[..., :self.C]                              # (batch_size, S, S, C)
        pred_boxes = predictions[..., self.C:] \
                        .reshape(-1, self.S, self.S, self.B, 5)             # (batch_size, S, S, B, 5)
        
        pred_boxes_coord = pred_boxes[..., :4]                              # (batch_size, S, S, B, 4)
        pred_boxes_confidence = pred_boxes[..., 4:5]                        # (batch_size, S, S, B, 1)
        
        pred_boxes_coord_copy = pred_boxes_coord.clone()
        pred_boxes_coord_copy[..., 2:] *= self.S                            # Convert width and height to relative to grid cell
        
        target_box_copy = target_box_coord.clone()
        target_box_copy[..., 2:] *= self.S                                  # Convert width and height to relative to grid cell
        target_box_copy = target_box_copy \
                    .unsqueeze(-2) \
                    .expand(-1, -1, -1, self.B, -1)                         # (batch_size, S, S, 4) -> (batch_size, S, S, B, 4)

        iou = intersection_over_union(
            pred_boxes_coord_copy,
            target_box_copy
        ).squeeze(dim=-1)                                                   # (batch_size, S, S, B)
        best_box_id = iou.argmax(dim=-1, keepdim=True)                      # (batch_size, S, S, 1)

        best_box_coord = torch.gather(pred_boxes_coord, dim=3, index=best_box_id.unsqueeze(-1).expand(-1, -1, -1, -1, 4)).squeeze(3)
        best_box_confidence = torch.gather(pred_boxes_confidence, dim=3, index=best_box_id.unsqueeze(-1)).squeeze(3)

        assert best_box_coord.shape == target_box_coord.shape
        assert best_box_confidence.shape == target_confidence.shape

        coord_loss = torch.sum(
            target_confidence * (
                torch.square(target_box_coord[..., :2] - best_box_coord[..., :2]) +
                torch.square(torch.sqrt(target_box_coord[..., 2:]) - torch.sqrt(best_box_coord[..., 2:].clamp(min=0)))
            ),
            dim=[1, 2, 3]
        )
        
        obj_loss = torch.sum(
            target_confidence * torch.square(1 - best_box_confidence),
            dim=[1, 2, 3]
        )

        noobj_loss = torch.sum(
            (1 - target_confidence).unsqueeze(-2) * torch.square(pred_boxes_confidence),
            dim=[1, 2, 3, 4]
        )

        cls_loss = torch.sum(
            target_confidence * torch.square(target_class - pred_class),
            dim=[1, 2, 3]
        )

        total_loss = self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * noobj_loss + cls_loss

        return total_loss.mean()

if __name__ == "__main__":
    batch_size = 64
    S = 7
    B = 2
    C = 80
    criterion = YOLOLoss(S=S, B=B, C=C)

    targets = torch.rand((batch_size, S, S, C + 5))
    predictions = torch.rand((batch_size, S, S, C + 5 * B))

    loss = criterion(predictions, targets)

    print(loss)