import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights

import config


def build_model(num_classes=None, use_imagenet_backbone=None, use_coco=False):
    """
    Faster R-CNN with ResNet50-FPN backbone.

    use_coco=True  → COCO_V1 pretrained full model, detection head replaced for num_classes
    use_coco=False → weights=None (no COCO), optional ImageNet backbone
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if use_imagenet_backbone is None:
        use_imagenet_backbone = config.USE_IMAGENET_BACKBONE

    if use_coco:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            min_size=416,
            max_size=416,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print(f"[Model] Faster R-CNN | {num_classes} classes | COCO pretrained | head replaced")
    else:
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if use_imagenet_backbone else None
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            min_size=416,
            max_size=416,
        )
        mode = "ImageNet backbone" if use_imagenet_backbone else "from scratch"
        print(f"[Model] Faster R-CNN | {num_classes} classes | {mode} | NO COCO")

    return model
