# Date Tue Nov 14 15:20:36 CST 2023
# by ChanMo
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
# https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn

import pathlib
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2
from torchvision import datasets, models
from torch.utils.tensorboard import SummaryWriter

from coco_eval import CocoEvaluator
import utils

torch.cuda.empty_cache()
writer = SummaryWriter()

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


def get_transform(train):
    transforms = [v2.ToImage()]
    transforms.append(v2.RandomIoUCrop())
    transforms.append(v2.SanitizeBoundingBoxes())
    if train:
        transforms.append(v2.RandomHorizontalFlip(0.5))
        transforms.append(v2.RandomPhotometricDistort(p=1))

    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.ToPureTensor())
    return v2.Compose(transforms)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# use our dataset and defined transformations
dataset_train = datasets.CocoDetection(
    './data/train/images/',
    './data/train/annotation-small.json',
    transforms=get_transform(True)
)
dataset_train = datasets.wrap_dataset_for_transforms_v2(dataset_train, target_keys=("boxes", "labels", "masks","image_id"))

dataset_val = datasets.CocoDetection(
    './data/val/images/',
    './data/val/annotation-small.json',
    transforms=get_transform(False)
)
dataset_val = datasets.wrap_dataset_for_transforms_v2(dataset_val, target_keys=("boxes", "labels", "masks","image_id"))

# indices_train = torch.randperm(len(dataset_train)).tolist()
# dataset_train = torch.utils.data.Subset(dataset_train, indices_train[:1000])
# indices_val = torch.randperm(len(dataset_val)).tolist()
# dataset_val = torch.utils.data.Subset(dataset_val, indices_val[-200:]).dataset

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=1,
    shuffle=True,
    # num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

# get the model using our helper function
model = get_model_instance_segmentation(101)
# model = models.get_model("maskrcnn_resnet50_fpn", weights=None, weights_backbone=None, num_classes=101)
model.to(device)


def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # loss_fn
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()

        writer.add_scalar('Train', losses_reduced, batch)

        if batch % 100 == 0:
            loss = losses_reduced
            current = (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    # test_loss, correct = 0, 0
    coco = dataloader.dataset.coco
    coco_evaluator = CocoEvaluator(coco, ['bbox', 'segm'])
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)
# optimizer = torch.optim.Adam(
#     params,
#     lr=1e-3,
# )


# and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=100, #3
#     gamma=0.1
# )

lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor = 1e-3,
    total_iters = len(dataset_train) - 1,
    verbose=False
)

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(data_loader_train, model, optimizer)
    lr_scheduler.step()
    test(data_loader_val, model)
    # torch.save(model.state_dict(), f'maskcrnn_{t}.pth')

writer.close()
torch.save(model.state_dict(), 'maskcrnn.pth')
print("Done!")
