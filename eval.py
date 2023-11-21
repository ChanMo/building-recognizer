import torch
import torchvision
from pycocotools.cocoeval import COCOEval


dataset = torchvision.datasets.CocoDetection(
    './data/val/images/',
    './data/val/annotation-small.json'
)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)

model = torchvision.models.get_model('maskrcnn_resnet50_fpn', num_classes=101)
model.load_state_dict(torch.load('./maskrcnn.pth'))

model.eval()
with torch.no_grad():
    for images, targets in dataloader:
        images = list(img.to('cuda') for img in images)
        outputs = model(images)
        outputs = [{k:v.to('cpu') for k, v in t.items()} for t in outputs]
        coco_gt = dataloader.dataset.coco
        e = COCOEval(
            coco_gt,
            COCO.loadRes(coco_gt, [{
                'image_id': row['image_id'],
                'category_id': 100,
                'segmentation': [],
                'score': 0
            } for row in outputs]),
            'segm'
        )
        e.evaluate()
        e.assumulate()
        e.summarize()
