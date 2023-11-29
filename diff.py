import pathlib
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.ops import nms
from utils import diff_boxes_list

zoom = 18
threshold = 0.68
model_name = 'maskrcnn_b5'
source_path = pathlib.Path(f'/home/chen/test_10/{zoom}/')
source_images = sorted(source_path.glob('**/*.png'))

# rows = len(source_path.iterdir())
cols = len(list(list(source_path.iterdir())[0].iterdir()))

model = torchvision.models.get_model('maskrcnn_resnet50_fpn', num_classes=3)
model.load_state_dict(torch.load(f'./{model_name}.pth'))
model.eval()

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    v2.ToPureTensor()
])

with torch.no_grad():
    res = []
    for index, img_path in enumerate(source_images):
        images = [
            read_image(str(img_path)),
            read_image(str(img_path).replace('_10/', '_03/'))
        ]
        images = [transforms(img[:3,...]) for img in images]
        outputs = model(images)
        boxes_list = []
        print(f'Image: {img_path}, Index: {index+1}/{len(source_images)}')
        for index, output in enumerate(outputs):
            masks = output['masks'].squeeze()
            if len(masks.shape) == 2:
                masks = masks.unsqueeze(0)

            # step 1
            masks = masks > threshold

            # step 2
            mask_areas = torch.sum(masks, dim=(1,2))
            keep = mask_areas > 500
            boxes = output['boxes'][keep]
            scores = output['scores'][keep]

            # step 3
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = box_areas < 80*80

            # step 4
            keep2 = nms(boxes[keep], scores[keep], 0.2)

            boxes = boxes[keep][keep2]
            boxes_list.append(boxes)

        boxes = diff_boxes_list(boxes_list)
        img = F.to_image(images[0])
        img = F.to_dtype(img, torch.uint8, scale=True)
        if isinstance(boxes, torch.Tensor):
            print(f'Find boxes: {boxes.shape}, Index: {index}')
            img = draw_bounding_boxes(img, boxes, colors='orange', width=4)

        im = F.to_dtype(img, torch.float32, scale=True)
        im = F.rotate(im, -90)
        res.append(im)

        grid = torchvision.utils.make_grid(res, cols, 0)
        torchvision.utils.save_image(F.rotate(grid, 90), 'output.png')
