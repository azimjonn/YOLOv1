import os
import tqdm
import torch
import config

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision.transforms.functional import to_tensor, normalize, resize, to_pil_image

def make_yolo_target(image, bboxes):
    '''
    Convets list of bounding boxes to yolo target format.

    Parameters:
    image: torch.Tensor, only used to get shapes of the image
    bboxes: dict, list of bounding boxes
        bbox: (x, y, width, height) - top left coordinate of image, width and height in image pixel coordinate system
        class: integer in range [0, C), object class index in bounding box

    Returns:
    Tensor: shape with (batch_size, S, S, C + 5 * B)
    '''
    assert image.shape[1] % config.S == 0
    assert image.shape[2] % config.S == 0

    grid_size_x = image.shape[1] / config.S
    grid_size_y = image.shape[2] / config.S

    depth = 5 + config.C
    target = torch.zeros((config.S, config.S, depth))
    
    patch_class = {}    # cell -> class

    for bbox in bboxes:
        x_min, y_min, width, height = bbox['bbox']
        class_idx = bbox['class']

        x_mid = x_min + width / 2
        y_mid = y_min + height / 2

        col = int(x_mid // grid_size_x)
        row = int(y_mid // grid_size_y)

        assert 0 <= col < config.S
        assert 0 <= row < config.S

        cell = (row, col)
        # one patch can only represent one class
        # and only one ground truth bounding box
        if cell not in patch_class:
            target[row, col, class_idx] = 1.0

            gt_bbox = (
                (x_mid - col * grid_size_x) / grid_size_x,
                (y_mid - row * grid_size_y) / grid_size_y,
                width / config.IMAGE_SIZE[0],
                height / config.IMAGE_SIZE[1],
                1.0
            )

            target[row, col, config.C:] = torch.tensor(gt_bbox)

            patch_class[cell] = class_idx

    return target
        
class CocoObjectDetection:
    def __init__(self, root, annFile, transforms=None, is_test=False):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.is_test = is_test

        self.category_to_class = {category_id:idx for idx, category_id in enumerate(self.coco.getCatIds())}
        self.class_to_category = {val:key for key, val in self.category_to_class.items()}
    
    def _load_image(self, id):
        file_name = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, file_name)

        return to_tensor(Image.open(path).convert('RGB'))
    
    def _load_target(self, id):
        if self.is_test:
            return []
        all_anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        detecton_anns = [
                {'bbox': ann['bbox'], 'class': self.category_to_class[ann['category_id']]}
                    for ann in all_anns
            ]
        return detecton_anns
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        bboxes = self._load_target(id)
        image_shape = image.shape

        if self.transforms is not None:
            image, bboxes = self.transforms(image, bboxes)
        
        # resize image and bbox
        scale_x = config.IMAGE_SIZE[1] / image.shape[2]
        scale_y = config.IMAGE_SIZE[0] / image.shape[1]
        
        image = resize(image, config.IMAGE_SIZE)
        resized_bboxes = []
        for bbox in bboxes:
            b = bbox['bbox']
            bbox['bbox'] = [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
            resized_bboxes.append(bbox)

        # normalize image
        image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if self.is_test:
            return image, (id, (image_shape[1], image_shape[2]))

        # create yolo target tensor
        target = make_yolo_target(image, resized_bboxes)
        
        return image, target
    
    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    dataset = CocoObjectDetection('/data/coco/images/val2017', '/data/coco/annotations/instances_val2017.json')

    for image, target in dataset:
        print(target.shape)