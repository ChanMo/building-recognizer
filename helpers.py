import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops import masks_to_boxes


def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            # if boxes is not None:
            #     # print(boxes.shape)
            #     # print(boxes)
            #     # for index, score in enumerate(target['scores']):
            #     #     if score > 0.5:
            #     #         img = draw_bounding_boxes(img, boxes[index], colors='yellow', width=2)
            #     # img = draw_bounding_boxes(img, boxes, colors="yellow", width=2)
            #     # boxes = boxes[target['scores'] > 0.8]
            #     # print(boxes.shape)
            #     # print(boxes)
            #     img = draw_bounding_boxes(img, boxes, colors="yellow", width=2)
            if masks is not None:
                # img = draw_segmentation_masks(img, (masks > 0.5).squeeze(1).to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)
                masks = masks > 0.5
                img = draw_segmentation_masks(img, masks, colors='green', alpha=0.65)
                boxes = masks_to_boxes(masks)
                img = draw_bounding_boxes(img, boxes, colors='red')

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
