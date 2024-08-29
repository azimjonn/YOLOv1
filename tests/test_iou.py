import unittest
import torch

from utils import intersection_over_union

class TestIntersectionOverUnion(unittest.TestCase):

    def test_identical_boxes(self):
        box1 = torch.tensor([0.5, 0.5, 0.2, 0.2])  # Centered in the grid, 20% of image width and height
        box2 = torch.tensor([0.5, 0.5, 0.2, 0.2])
        iou = intersection_over_union(box1, box2)
        self.assertAlmostEqual(iou.item(), 1.0, places=6)

    def test_non_overlapping_boxes(self):
        box1 = torch.tensor([0.1, 0.1, 0.02, 0.02])  # Near the top-left corner
        box2 = torch.tensor([0.9, 0.9, 0.02, 0.02])  # Near the bottom-right corner
        iou = intersection_over_union(box1, box2)
        self.assertEqual(iou.item(), 0.0)

    def test_partial_overlap(self):
        box1 = torch.tensor([0.4, 0.4, 0.4, 0.4])  # Large box centered near (0.4, 0.4)
        box2 = torch.tensor([0.6, 0.6, 0.4, 0.4])  # Large box centered near (0.6, 0.6)
        iou = intersection_over_union(box1, box2)
        expected_iou = 0.142857  # Expected IoU calculated based on overlap
        self.assertAlmostEqual(iou.item(), expected_iou, places=6)

    def test_edge_touching_boxes(self):
        box1 = torch.tensor([0.25, 0.5, 0.5, 0.5])  # Wide box centered at (0.25, 0.5)
        box2 = torch.tensor([0.75, 0.5, 0.5, 0.5])  # Wide box centered at (0.75, 0.5)
        iou = intersection_over_union(box1, box2)
        self.assertEqual(iou.item(), 0.0)

    def test_box_with_zero_area(self):
        box1 = torch.tensor([0.5, 0.5, 0.0, 0.0])  # Zero area box
        box2 = torch.tensor([0.5, 0.5, 0.2, 0.2])  # Regular box
        iou = intersection_over_union(box1, box2)
        self.assertEqual(iou.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
