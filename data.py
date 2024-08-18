import os
import tqdm

from PIL import Image
from pycocotools.coco import COCO

class CocoObjectDetection:
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

        self.category_to_class = {category_id:idx for idx, category_id in enumerate(self.coco.getCatIds())}
        self.class_to_category = {val:key for key, val in self.category_to_class.items()}
    
    def _load_image(self, id):
        file_name = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, file_name)

        return Image.open(path).convert('RGB')
    
    def _load_target(self, id):
        all_anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        detecton_anns = [
                {'bbox': ann['bbox'], 'class': self.category_to_class[ann['category_id']]}
                    for ann in all_anns
            ]
        return detecton_anns
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def __len__(self):
        return len(self.ids)