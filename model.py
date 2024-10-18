import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class FasterRCNNResNet50(torch.nn.Module):
    def __init__(self, num_classes=14, nms_iou_threshold=0.5, score_threshold=0.05):
        super(FasterRCNNResNet50, self).__init__()
        
        # Backbone with FPN
        backbone = resnet_fpn_backbone('resnet50', weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze early layers to focus on fine-tuning higher-level features
        for i, layer in enumerate(backbone.body.children()):
            if i < 6:
                for param in layer.parameters():
                    param.requires_grad = False 

        # RoI Pooler with modified settings
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,  
            sampling_ratio=4  # Higher sampling ratio for better feature extraction
        )

        # Faster R-CNN model definition
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            box_roi_pool=roi_pooler
        )

        # Replace the box predictor to match the number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        # Store NMS parameters
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold

    def forward(self, images, targets=None):
        """
        Forward method for the Faster R-CNN model.
        
        Args:
            images (Tensor): Images for the model.
            targets (list): List of target dictionaries for training (optional).
        
        Returns:
            output (Tensor): Model outputs during inference.
        """
        # Get the raw predictions from the model
        outputs = self.model(images, targets)
        
        if self.training:
            # During training, return the raw output (losses)
            return outputs
        
        # Apply NMS during evaluation
        filtered_outputs = []
        for output in outputs:
            boxes = output['boxes']
            scores = output['scores']
            labels = output['labels']

            # Filter out low-confidence scores
            high_score_idx = scores > self.score_threshold
            boxes = boxes[high_score_idx]
            scores = scores[high_score_idx]
            labels = labels[high_score_idx]

            # Apply NMS
            nms_idx = torchvision.ops.nms(boxes, scores, self.nms_iou_threshold)
            boxes = boxes[nms_idx]
            scores = scores[nms_idx]
            labels = labels[nms_idx]

            # Store the filtered results
            filtered_outputs.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })

        return filtered_outputs
