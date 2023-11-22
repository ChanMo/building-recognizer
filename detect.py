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

parser = argparse.ArgumentParser(
    prog='Building Detect',
    description='detech buildlings from an image'
)
parser.add_argument('filename')
parser.add_argument('--model', default='maskrcnn_b1')
parser.add_argument('--threshold', default=0.68, type=float)
args = parser.parse_args()

for file in os.listdir('./outputs/'):
    os.remove('./outputs/' + file)

model = torchvision.models.get_model('maskrcnn_resnet50_fpn', num_classes=101)
model.load_state_dict(torch.load(f'./{args.model}.pth'))

def read_images(filename):
    return [
        read_image(f'/home/chen/test_03/{args.filename}')[:3,...],
        read_image(f'/home/chen/test_10/{args.filename}')[:3,...],
    ]

def diff_boxes(source, target):
    # find the difference from a boxes
    iou = torchvision.ops.box_iou(source, target)
    diffes = []
    diffes = iou[0]
    for i,row in enumerate(iou):
        for j,col in enumerate(row):
            if col != 0:
                diffes[j] = 1

    return diffes == 0

def diff_boxes_list(boxes_list):
    boxes1 = boxes_list[0]
    boxes2 = boxes_list[1]
    res = [
        boxes1[diff_boxes(boxes2, boxes1)],
        boxes2[diff_boxes(boxes1, boxes2)]
    ]
    return torch.cat(res)

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
        boxes = boxes[mask_areas > 100]
        scores = scores[mask_areas > 100]

        # Thresholding based on area
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = box_areas < 100*100

        # Absolute size thresholding
        # max_side_length = 200
        # keep = torch.logical_and(boxes[:, 2] < max_side_length, boxes[:, 3] < max_side_length)

        # NMS
        # keep = nms(boxes, scores, 0.2)

        boxes = boxes[keep]
        scores = scores[keep]

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
