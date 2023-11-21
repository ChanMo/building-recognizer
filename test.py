import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision import datasets, tv_tensors
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.functional as F
from helpers import plot



ROOT = pathlib.Path('./data/train/')
IMAGES_PATH = str(ROOT / 'images')
ANNOTATIONS_PATH = str(ROOT / 'annotation-small.json')

dataset = datasets.CocoDetection(
    IMAGES_PATH,
    ANNOTATIONS_PATH,
    transforms=v2.Compose([
        v2.ToImage(),
        v2.SanitizeBoundingBoxes(),
    ])
)
dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "masks"))
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:4])

plot([[dataset[0], dataset[1]], [dataset[2], dataset[3]]])
plt.savefig('output.png')
