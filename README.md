# Weighted Mask Fusion

This repository contains a Python implementation of Weighted Masks Fusion (WMF), which is a method for ensembling predicted masks from semantic or instance segmentation models. The WMF algorithm is inspirated by Weighted Boxes Fusion (WBF) introduced by ZFTurbo. It utilizes the principle of weighted fusion to combine masks from different segmentation models based on their confidence scores and other configurable parameters (e.g., conf_type, soft_weight, iou_thr, model_weights, etc.). The fused masks, scores, and boxes are used to generate an ensemble output with improved segmentation performance compared to individual models. The original code and concept for Weighted Masks Fusion were contributed by Kaggler Odede, whose solution achieved the 8th position in the "[Sartorius - Cell Instance Segmentation](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation)" Kaggle competition. 


## Prerequisites

Make sure you have the following prerequisites installed before using this library:

- Python 3
- Numba
- NumPy

## Installation

You can install Weighted Mask Fusion by cloning this repository:

```bash
git clone https://github.com/chrise96/Weighted-Masks-Fusion.git
```

## Example: Weighted Masks Fusion

The following code example demonstrates how to use the `weighted_masks_fusion` function to perform ensemble and fusion of masks from multiple YOLOv8 instance segmentation models.

```python
inmodels = []
# Loop through all YOLOv8 models and perform predictions
for i, model in enumerate(yolo_models):
    pred_boxes, pred_masks, scores, pred_classes = yolov8_predict(model, img)
    inmodels += [i] * len(scores)
    
    # Combine predictions from multiple models
    if i == 0:
        all_pred_boxes = pred_boxes
        all_pred_masks = pred_masks
        all_scores = scores
    else:
        all_pred_boxes = np.vstack([all_pred_boxes, pred_boxes])
        all_pred_masks = np.vstack([all_pred_masks, pred_masks])
        all_scores = np.hstack([all_scores, scores])

# Perform Weighted Masks Fusion on the combined predictions
pred_masks, scores, pred_boxes = weighted_masks_fusion(all_pred_masks, all_pred_boxes, all_scores, inmodels,
                                           num_models=len(yolo_models), conf_type='model_weight', model_weights=[1,1,1]
                                           )
```

## References

- Weighted Boxes Fusion (WBF) for use in object detection. [Link](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- Weighted Segments Fusion (WSF) for use in instance segmentation. [Link](https://www.kaggle.com/code/mistag/sartorius-tta-with-weighted-segments-fusion)
- Ensemble boxes implementations in the 8th place solution shared by Kaggler Odede. [Link](https://www.kaggle.com/code/markunys/8th-place-solution-inference)
