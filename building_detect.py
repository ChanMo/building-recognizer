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

def read_images(filename):
    return [
        read_image(f'/home/chen/test_03/{args.filename}')[:3,...],
        read_image(f'/home/chen/test_10/{args.filename}')[:3,...],
    ]

images = read_images(args.filename)

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

    _, axs = plt.subplots(ncols=3, nrows=len(images), figsize=(8,8))

    boxes_list = []
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
        # keep = box_areas < 120*120 # 19
        keep = box_areas < 80*80 # 18
        # keep = box_areas < 50*50 # 17

        # Absolute size thresholding
        # max_side_length = 200
        # keep = torch.logical_and(boxes[:, 2] < max_side_length, boxes[:, 3] < max_side_length)

        # NMS
        keep2 = nms(boxes[keep], scores[keep], 0.1)
        boxes = boxes[keep][keep2]
        scores = scores[keep][keep2]

        boxes_list.append(boxes)
        # for index, box in enumerate(boxes[keep]):
        #     box_img = F.crop(img, box[1].long(), box[0].long(), (box[3]-box[1]).long(), (box[2]-box[0]).long())
        #     box_img = F.to_pil_image(box_img)
        #     box_img.save(f'outputs/{index}.png')

        labels = [f"S:{score:.2f}" for score in scores]
        masks_img = draw_segmentation_masks(img, masks, colors="green", alpha=.65)
        boxes_img = draw_bounding_boxes(img, boxes, labels, colors='red', font='arial.ttf', font_size=14)

        axs[index, 1].imshow(masks_img.permute(1,2,0).numpy())
        axs[index, 0].imshow(boxes_img.permute(1,2,0).numpy())


    boxes = diff_boxes_list(boxes_list)
    if isinstance(boxes, torch.Tensor):
        print(boxes.shape)
        for index in range(len(outputs)):
            img = F.to_image(images[index])
            img = F.to_dtype(img, torch.uint8, scale=True)
            img = draw_bounding_boxes(img, boxes, colors='blue')
            axs[index, 2].imshow(img.permute(1,2,0).numpy())

    plt.suptitle(args.filename)
    plt.savefig('output.png')
