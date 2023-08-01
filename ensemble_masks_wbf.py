import warnings
import numpy as np
from numba import jit

@jit(nopython=True)
def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_weighted_mask(masks, scores, inmodels, conf_type):
    mask = np.zeros(masks[0].shape, dtype=np.float32)
    conf = 0
    conf_list = []
    for m, s, im in zip(masks, scores, inmodels):
        if conf_type == 'model_weight2':
            mask += s * im * m
            conf += s * im
        else:
            mask += s * m
            conf += s
        conf_list.append(s)
    score = np.max(conf_list)
    mask = mask / conf
    return mask, score, conf_list


def get_weighted_box(boxes, scores, inmodels, conf_type):
    box = np.zeros(4, dtype=np.float32)
    conf = 0
    conf_list = []
    for b, s, im in zip(boxes, scores, inmodels):
        if conf_type == 'model_weight2':
            box += s * im * b
            conf += s * im
        else:
            box += s * b
            conf += s
        conf_list.append(s)
    score = np.max(conf_list)
    box = box / conf
    return box, score


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        iou = bb_intersection_over_union(box, new_box)
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_masks_fusion(masks, boxes, scores, models, iou_thr=0.7, skip_mask_thr=0.0, 
                          conf_type='max_weight', soft_weight=5, thresh_type=None, model_weights=1,
                          num_thresh=4, num_models=5):
    masks = masks[scores > skip_mask_thr]
    boxes = boxes[scores > skip_mask_thr]
    models = models[scores > skip_mask_thr]
    scores = scores[scores > skip_mask_thr]
    
    new_masks = []
    new_boxes = []
    new_scores = []
    inmodels = []
    weighted_boxes = []
    weighted_scores = []
    # Clusterize boxes
    for i in range(len(masks)):
            
        index, best_iou = find_matching_box(weighted_boxes, boxes[i], iou_thr)
        if index != -1:
            new_masks[index].append(masks[i])
            new_boxes[index].append(boxes[i])
            new_scores[index].append(scores[i])
            inmodels[index].append(models[i])
            weighted_boxes[index], weighted_scores[index] = get_weighted_box(new_boxes[index], new_scores[index], inmodels[index], conf_type)
        else:
            new_masks.append([masks[i]])
            new_boxes.append([boxes[i].copy()])
            new_scores.append([scores[i].copy()])
            inmodels.append([models[i]])
            weighted_boxes.append(boxes[i].copy())
            weighted_scores.append(scores[i].copy())
            
    ens_masks = []
    ens_scores = []
    ens_boxes = []
    for nmasks, nscores, wbox, inms in zip(new_masks, new_scores, weighted_boxes, inmodels):
        mask, score, conf_list = get_weighted_mask(nmasks, nscores, inms, conf_type)
        if thresh_type == 'num_thresh':
            if len(conf_list) >= num_thresh:
                ens_masks.append(mask)
                ens_boxes.append(wbox)
            else:
                continue
        else:
            ens_masks.append(mask)
            ens_boxes.append(wbox)

        if conf_type == 'max_weight':
            ens_scores.append(score * min(len(conf_list), num_models) / num_models)
        elif conf_type == 'max':
            ens_scores.append(score)
        elif conf_type == 'soft_weight':
            ens_scores.append(score * (min(len(conf_list), num_models) + soft_weight) / (soft_weight + num_models))
        elif conf_type == 'model_weight' or conf_type == 'model_weight2':
            this_weights = [model_weights[i] for i in inms]
            ens_scores.append(score * (min(np.sum(this_weights), np.sum(model_weights)) + soft_weight) / (soft_weight + np.sum(model_weights)))

    return ens_masks, ens_scores, ens_boxes
