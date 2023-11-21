import os
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
parser.add_argument('--model', default='maskcrnn')
# parser.add_argument('--show-masks', default=False, type=bool)
parser.add_argument('--zoom', default=19, type=int)
args = parser.parse_args()

for file in os.listdir('./outputs/'):
    os.remove('./outputs/' + file)

model = torchvision.models.get_model('maskrcnn_resnet50_fpn', num_classes=101)
model.load_state_dict(torch.load(f'./{args.model}.pth'))

def read_images(filename, zoom):
    return [
        read_image(f'/home/chen/test_03/{zoom}/{args.filename}')[:3,...],
        read_image(f'/home/chen/test_10/{zoom}/{args.filename}')[:3,...],
    ]

images = read_images(args.filename, args.zoom)

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

    _, axs = plt.subplots(ncols=len(images), nrows=2)

    for index, output in enumerate(outputs):
        img = F.to_image(images[index])
        img = F.to_dtype(img, torch.uint8, scale=True)

        masks = output['masks']
        masks = masks.squeeze()
        # keep = torch.amax(masks > 0.68, dim=(1,2))
        keep = torch.amax(masks > 0.58, dim=(1,2))
        boxes = output['boxes']

        # for index, box in enumerate(boxes[keep]):
        #     box_img = F.crop(img, box[1].long(), box[0].long(), (box[3]-box[1]).long(), (box[2]-box[0]).long())
        #     box_img = F.to_pil_image(box_img)
        #     box_img.save(f'outputs/{index}.png')

        labels = [f"S:{score:.2f}" for score in output["scores"][keep]]
        # if args.show_masks:
        masks_img = draw_segmentation_masks(img, masks > 0.4, colors="green", alpha=.65)
        img = draw_bounding_boxes(img, boxes[keep], labels, colors='red', font='arial.ttf', font_size=14)

        axs[index, 0].imshow(img.permute(1,2,0).numpy())
        axs[index, 1].imshow(masks_img.permute(1,2,0).numpy())

    plt.savefig('output.png')
