import torch
from torchvision.datasets import CocoDetection

d = CocoDetection('/home/chen/Code/data/crowdai-mapping-challenge/val/images/', '/home/chen/Code/data/crowdai-mapping-challenge/val/annotation.json')

print(len(d))
print(d[0])
