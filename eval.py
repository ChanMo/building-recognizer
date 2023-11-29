import argparse
import torch
import torchvision
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2

from coco_eval import CocoEvaluator
from coco_utils import convert_to_coco_api

parser = argparse.ArgumentParser(
    prog='Model Evaluation'
)
parser.add_argument('--model', default='maskrcnn_b1')
args = parser.parse_args()

transforms = v2.Compose([
    v2.ToImage(),
    # v2.RandomIoUCrop(),
    # v2.SanitizeBoundingBoxes(),
    v2.ToDtype(torch.float, scale=True),
    v2.ToPureTensor(),
])

dataset = CocoDetection(
    # './data/val/images/',
    # './data/val/annotation-small.json',
    '/home/chen/Code/coco-annotator/datasets/Buildings10/',
    './data/Buildings10-5.json',
    transforms=transforms
)
dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "masks","image_id", "area", "iscrowd"))
dataset = torch.utils.data.Subset(dataset, torch.randperm(100).tolist())
dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=False,
    batch_size=1,
    collate_fn=lambda batch: tuple(zip(*batch))
)

model = torchvision.models.get_model('maskrcnn_resnet50_fpn', num_classes=101)
model.load_state_dict(torch.load(f'./{args.model}.pth'))
model.to('cuda')

model.eval()
coco = convert_to_coco_api(dataloader.dataset.dataset)
# coco = dataloader.dataset.dataset.coco
coco_evaluator = CocoEvaluator(coco, ['bbox']) # segm
with torch.no_grad():
    for images, targets in dataloader:
        images = list(img.to('cuda') for img in images)
        outputs = model(images)
        outputs = [{k:v.to('cpu') for k, v in t.items()} for t in outputs]
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()


