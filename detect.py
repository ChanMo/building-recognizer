import sys
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


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


model = get_model_instance_segmentation(101)
model.load_state_dict(torch.load('./maskcrnn.pth'))

image = read_image(args.filename)

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    #v2.ToPureTensor()
])


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.eval()
with torch.no_grad():
    img = transforms(image)
    # x = x[:3, ...].to(device)
    #x = x.to(device)
    output = model([img])[0]

    plot([(img,output)])
    plt.savefig('output.png')
    sys.exit()
    #img = (255.0 * (img - img.min()) / (img.max() - img.min())).to(torch.uint8)
    img = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
    ])(img)

    score_threshold = args.score_threshold
    #boxes = output['boxes'].long()
    #img = draw_bounding_boxes(img, boxes=boxes[output['scores'] > score_threshold], width=2)
    masks = (output["masks"] > 0.5).squeeze(1)
    # masks = output["masks"] > 0.5
    img = draw_segmentation_masks(img, masks, alpha=0.5, colors="blue")
    boxes = masks_to_boxes(masks)
    img = draw_bounding_boxes(img, boxes, colors='red')

    #plt.figure(figsize=(12, 12))
    #plt.imshow(output_image.permute(1, 2, 0))
    #plt.imsave('output.png', output_image.permute(1, 2, 0))
    img = F.to_pil_image(img)
    #img = output_image.detach()
    #plt.imsave('output.png', img)
    plt.imshow(np.asarray(img))
    plt.savefig('output.png')
