# Building Recognizer

建筑物识别器是一种用于识别卫星图或航拍图中的建筑物的模型。该模型使用卷积神经网络 (CNN) 来提取图像的特征，然后使用分类器来识别图像中的建筑物

该模型的输入数据是卫星图或航拍图。输出数据是图像中建筑物的类别和边界框。

![output.png](output.png)

该模型可以用于各种应用，例如：

* 城市规划：用于识别城市中的建筑物，以进行规划和管理。
* 灾害管理：用于识别灾害发生后受损的建筑物，以进行救援和重建。
* 环境监测：用于识别环境中的建筑物，以进行监测和保护。

建筑物识别器是一项具有重要应用价值的技术。随着卫星图像和航拍图像技术的不断发展，建筑物识别器将在未来得到更加广泛的应用。


## 模型选择

模型使用的是Mask-RCNN
```
model = torchvision.models.detection.maskrcnn_resnet50_fpn()

```

## 数据集

数据集使用的是CrowdAI的MappingChallenge中的数据集, 此数据集使用和COCO相同的格式, 可以直接通过CocoDetection加载
```
dataset = datasets.CocoDetection(
    IMAGES_PATH,
    ANNOTATIONS_PATH,
    ...
)
```

## 为模型增加新的特征值

数据集中建筑物的Label是100
```
num_classes = 101
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

## Reference:

* https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
* https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn
* https://www.aicrowd.com/challenges/mapping-challenge/dataset_files
