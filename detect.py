import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms
from torchvision.transforms.v2 import functional as F
from helpers import plot
from utils import diff_boxes_list

parser = argparse.ArgumentParser(
    prog='Building Detect',
    description='detech buildlings from an image'
)
parser.add_argument('filename')
parser.add_argument('--model', default='maskrcnn') # maskrcnn_b5
parser.add_argument('--num-classes', default=101, type=int) # 3
parser.add_argument('--threshold', default=0.68, type=float)
args = parser.parse_args()

for file in os.listdir('./outputs/'):
    os.remove('./outputs/' + file)

model = torchvision.models.get_model('maskrcnn_resnet50_fpn', num_classes=args.num_classes) #101
model.load_state_dict(torch.load(f'./{args.model}.pth'))

print(args.filename)

images = [read_image(args.filename)]

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    v2.ToPureTensor()
])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.eval()

with torch.no_grad():
    images = [transforms(img) for img in images]
    outputs = model(images)

    for index, output in enumerate(outputs):
        img = F.to_image(images[index])
        img = F.to_dtype(img, torch.uint8, scale=True)

        masks = output['masks'].squeeze()
        masks = masks > args.threshold

        boxes = output['boxes']
        scores = output['scores']

        mask_areas = torch.sum(masks, dim=(1,2))
        keep = mask_areas > 500
        boxes = boxes[keep]
        scores = scores[keep]

        # Thresholding based on area
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = box_areas < 200*200 # 18

        # Absolute size thresholding
        # max_side_length = 200
        # keep = torch.logical_and(boxes[:, 2] < max_side_length, boxes[:, 3] < max_side_length)

        # NMS
        keep2 = nms(boxes[keep], scores[keep], 0.1)
        boxes = boxes[keep][keep2]
        scores = scores[keep][keep2]

        labels = [f"S:{score:.2f}" for score in scores]
        img = draw_segmentation_masks(img, masks, colors="green", alpha=.65)
        img = draw_bounding_boxes(img, boxes, labels, colors='red', font='arial.ttf', font_size=14)

        plt.imshow(img.permute(1,2,0).numpy())

    plt.title(os.path.split(args.filename)[1])
    plt.savefig('output.png')
