import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as F
import torchvision
from helpers import plot

parser = argparse.ArgumentParser(
    prog='Building Detect',
    description='detech buildlings from an image'
)
parser.add_argument('filename')
parser.add_argument('-s', '--score_threshold', type=float, default=0.8)
args = parser.parse_args()

for file in os.listdir('./outputs/'):
    os.remove('./outputs/' + file)

model = torchvision.models.get_model('maskrcnn_resnet50_fpn', num_classes=101)
model.load_state_dict(torch.load('./maskcrnn.pth'))

image = read_image(args.filename)
image = image[:3,...]

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    v2.ToPureTensor()
])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.eval()

with torch.no_grad():
    img = transforms(image)
    output = model([img])[0]
    plot([(img,output)])
    plt.savefig('output.png')
