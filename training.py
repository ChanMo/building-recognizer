# Date Tue Nov 14 15:20:36 CST 2023
# by ChanMo
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
# https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn

import sys
import pathlib
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2
from torchvision import datasets, tv_tensors, models

from engine import train_one_epoch, evaluate
import utils

# Call `torch.cuda.empty_cache()`.
torch.cuda.empty_cache()

# init dataset
# https://www.aicrowd.com/challenges/mapping-challenge/dataset_files
ROOT = pathlib.Path('/home/chen/Code/data/crowdai-mapping-challenge/val/')
IMAGES_PATH = str(ROOT / 'images')
ANNOTATIONS_PATH = str(ROOT / 'annotation-small.json')


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    #print(model.roi_heads.box_predictor.cls_score.weight.shape)
    #print(model.roi_heads)
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
    transforms = []
    if train:
        transforms.append(v2.RandomHorizontalFlip(0.5))
    transforms.append(v2.ToDtype(torch.float, scale=True))
    transforms.append(v2.ToPureTensor())
    return v2.Compose(transforms)

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# train on the GPU or on the CPU, if a GPU is not available
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

# our dataset has one class only - buildling
num_classes = 101
# use our dataset and defined transformations
dataset = datasets.CocoDetection(
    IMAGES_PATH,
    ANNOTATIONS_PATH,
    transforms=transforms
    #get_transform(train=True)
)
dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "masks",))
img, target = dataset[0]
#print(target['labels'])

dataset_test = datasets.CocoDetection(
    IMAGES_PATH,
    ANNOTATIONS_PATH,
    transforms=transforms
    #get_transform(train=False)
)
dataset_test = datasets.wrap_dataset_for_transforms_v2(dataset_test, target_keys=("boxes", "labels", "masks",))
#

# # split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    # num_workers=4,
    # collate_fn=utils.collate_fn
    collate_fn=lambda batch: tuple(zip(*batch)),
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    # num_workers=4,
    # collate_fn=utils.collate_fn
    collate_fn=lambda batch: tuple(zip(*batch)),
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)
#sys.exit()
#model = models.get_model("maskrcnn_resnet50_fpn_v2", weights=None, weights_backbone=None)
model.to(device)

def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        #X, y = X.to(device), y.to(device)

        # Compute prediction error
        #print(images, targets)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        #loss = loss_fn(pred, y)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # Backpropagation
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            #loss, current = loss.item(), (batch + 1) * len(X)
            loss = losses_reduced
            current = (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for images, targets in dataloader:
#             #X, y = X.to(device), y.to(device)
#             images = list(img.to(device) for img in images)
#             outputs = model(images)
#             outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
#
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# move model to the right device

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it for 5 epochs
num_epochs = 5

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(data_loader, model, optimizer)
    #test(data_loader_test, model)
    evaluate(model, data_loader_test, device=device)


# for epoch in range(num_epochs):
#     # # train for one epoch, printing every 10 iterations
#     # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
#     # # update the learning rate
#     # lr_scheduler.step()
#     # # evaluate on the test dataset
#     # evaluate(model, data_loader_test, device=device)
#     pass

torch.save(model.state_dict(), 'maskcrnn.pth')
print("Done!")
