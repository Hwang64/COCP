import logging
import numpy as np
from collections import Counter

def eva_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap




def do_train_evaluation(
    predictions,
    ground_truth,
):
    logger = logging.getLogger("maskrcnn_benchmark.training_evaluation")

    cur_gt_bbox_num = 0
    cur_pre_bbox_num = 0
    pre_bbox = predictions[:,0:4]
    pre_score = predictions[:,4]
    pre_label = predictions[:,5].astype(np.int32)
    gt_bbox = ground_truth[:,0:4]
    gt_label = ground_truth[:,4]

    sort_ind = np.argsort(gt_label)
    gt_label = gt_label[sort_ind].astype(np.int32)
    gt_bbox = gt_bbox[sort_ind,:]

    gt_info = Counter(gt_label)
    pre_info = Counter(pre_label)

    #ap_list = []
    R = {}
    total_ap = 0
    pre_bbox_previous_num = 0
    pre_bbox_previous_dict = {}
    for pre_cls in pre_info.keys():
        pre_bbox_previous_num += pre_info[pre_cls]
        pre_bbox_previous_dict[pre_cls] = pre_bbox_previous_num

    for gt_cls in gt_info.keys():


        if gt_cls == 0:continue
        gt_bbox_num = gt_info[gt_cls]
        cls_gt_bbox = gt_bbox[cur_gt_bbox_num:cur_gt_bbox_num+gt_bbox_num,:]
        R['bbox'] = cls_gt_bbox
        R['det'] = np.zeros(gt_bbox_num).astype(np.int32)
        BBGT = R['bbox'].astype(np.float32)


        if  pre_info[gt_cls] == 0:
            continue

        pre_bbox_num  = pre_info[gt_cls]
        cur_pre_bbox_num = pre_bbox_previous_dict[gt_cls]
        cls_pre_bbox  = pre_bbox[cur_pre_bbox_num-pre_bbox_num:cur_pre_bbox_num,:]
        cls_pre_score = pre_score[cur_pre_bbox_num-pre_bbox_num:cur_pre_bbox_num]

        cls_pre_sort_ind = np.argsort(-cls_pre_score)
        bbs = cls_pre_bbox[cls_pre_sort_ind,:].astype(np.float32)

        nd = bbs.shape[0]
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if bbs.size > 0:
            # compute overlaps
            # intersection
            for d in range(nd):
                bb=bbs[d,:]
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                if ovmax > 0.5:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(gt_bbox_num)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = eva_ap(rec, prec, True)

        cur_gt_bbox_num += gt_bbox_num
        total_ap += ap

    return total_ap
