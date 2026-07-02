import numpy as np
import os
import nibabel as nib

from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Pixel-wise confusion per class
# -----------------------------
def compute_pixel_metrics(pred, gt, class_id):

    pred_bin = (pred == class_id)
    gt_bin   = (gt == class_id)

    TP = np.logical_and(pred_bin, gt_bin).sum()
    FP = np.logical_and(pred_bin, ~gt_bin).sum()
    FN = np.logical_and(~pred_bin, gt_bin).sum()

    return TP, FP, FN


# -----------------------------
# Confidence interval
# -----------------------------
def compute_ci(values):
    values = np.array(values)
    mean = np.mean(values)

    if len(values) < 2:
        return mean, 0.0

    std = np.std(values, ddof=1)
    ci = 1.96 * std / np.sqrt(len(values))
    return mean, ci





def calculate_metrics_per_pixel(pred_folder, gt_folder, classes, list_of_interest = None):

    # -----------------------------
    # Storage (per image)
    # -----------------------------

    # macro averaging
    per_image_metrics = {c: {"precision": [], "recall": [], "f1": [], "iou": [], "dice": []} for c in classes}

    # micro averaging
    #global_counts = {c: {"TP": 0, "FP": 0, "FN": 0} for c in classes}
    n_images = 0


    # -----------------------------
    # Loop over dataset
    # -----------------------------
    for file in os.listdir(pred_folder):

        if not file.endswith(".nii.gz"):
            continue
        
        case_name = file.replace(".nii.gz", "")
        if list_of_interest is not None and case_name not in list_of_interest:
            continue

        pred_path = os.path.join(pred_folder, file)
        gt_path   = os.path.join(gt_folder, file)

        if not os.path.exists(gt_path):
            print("Missing GT:", file)
            continue

        pred = nib.load(pred_path).get_fdata().astype(int)
        gt   = nib.load(gt_path).get_fdata().astype(int)

        n_images += 1

        for c in classes:

            TP, FP, FN = compute_pixel_metrics(pred, gt, c)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
            dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

            per_image_metrics[c]["precision"].append(precision)
            per_image_metrics[c]["recall"].append(recall)
            per_image_metrics[c]["f1"].append(f1)
            per_image_metrics[c]["iou"].append(iou)
            per_image_metrics[c]["dice"].append(dice)


    # -----------------------------
    # Final results
    # -----------------------------
    print(f"\nEvaluated on {n_images} images\n")
    print("Pixel-level metrics with 95% confidence intervals\n")

    mean_precisions = []
    mean_recalls = []
    mean_f1s = []
    mean_ious = []
    mean_dices = []

    for c in classes:

        p_mean, p_ci = compute_ci(per_image_metrics[c]["precision"])
        r_mean, r_ci = compute_ci(per_image_metrics[c]["recall"])
        f1_mean, f1_ci = compute_ci(per_image_metrics[c]["f1"])
        iou_mean, iou_ci = compute_ci(per_image_metrics[c]["iou"])
        dice_mean, dice_ci = compute_ci(per_image_metrics[c]["dice"])

        mean_precisions.append(p_mean)
        mean_recalls.append(r_mean)
        mean_f1s.append(f1_mean)
        mean_ious.append(iou_mean)
        mean_dices.append(dice_mean)

        print(f"Class {c}")
        print(f" Dice:      {dice_mean:.3f} ± {dice_ci:.3f}")
        print(f" IoU:       {iou_mean:.3f} ± {iou_ci:.3f}")
        print(f" Precision: {p_mean:.3f} ± {p_ci:.3f}")
        print(f" Recall:    {r_mean:.3f} ± {r_ci:.3f}")
        print(f" F1:        {f1_mean:.3f} ± {f1_ci:.3f}")
        print(
    f"& {dice_mean:.3f} $\\pm$ {dice_ci:.3f} "
    f"& {iou_mean:.3f} $\\pm$ {iou_ci:.3f} "
    f"& {p_mean:.3f} $\\pm$ {p_ci:.3f} "
    f"& {r_mean:.3f} $\\pm$ {r_ci:.3f} "
)

    print("Mean over classes")
    print(f" Dice:      {np.mean(mean_dices):.4f}")
    print(f" Dice:      {np.mean(mean_dices):.4f}")
    print(f" IoU:       {np.mean(mean_ious):.4f}")
    print(f" Precision: {np.mean(mean_precisions):.4f}")
    print(f" Recall:    {np.mean(mean_recalls):.4f}")
    print(f" F1:        {np.mean(mean_f1s):.4f}")




# -----------------------------
# IoU computation
# -----------------------------
def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return inter / union


# -----------------------------
# Structure metrics (Hungarian matching)
# -----------------------------
def structure_metrics(pred_mask, gt_mask, class_id, iou_threshold):

    pred_bin = pred_mask == class_id
    gt_bin   = gt_mask == class_id

    pred_labels, n_pred = label(pred_bin)
    gt_labels, n_gt     = label(gt_bin)

    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0

    iou_matrix = np.zeros((n_gt, n_pred))

    for g in range(1, n_gt + 1):
        gt_region = gt_labels == g

        for p in range(1, n_pred + 1):
            pred_region = pred_labels == p
            iou_matrix[g-1, p-1] = compute_iou(gt_region, pred_region)

    if iou_matrix.size == 0:
        return 0, n_pred, n_gt

    cost_matrix = 1 - iou_matrix
    gt_ind, pred_ind = linear_sum_assignment(cost_matrix)

    TP = 0
    for g, p in zip(gt_ind, pred_ind):
        if iou_matrix[g, p] >= iou_threshold:
            TP += 1

    FP = n_pred - TP
    FN = n_gt - TP

    return TP, FP, FN




def calculate_metrics_per_structure(pred_folder, gt_folder, classes, iou_threshold, list_of_interest = None):
    # -----------------------------
    # Dataset loop
    # -----------------------------
    per_image_metrics = {
        c: {"precision": [], "recall": [], "f1": [], "iou50": []}
        for c in classes
    }

    n_images = 0

    for file in os.listdir(pred_folder):

        if not file.endswith(".nii.gz"):
            continue

        case_name = file.replace(".nii.gz", "")
        if list_of_interest is not None and case_name not in list_of_interest:
            continue
    
        pred_path = os.path.join(pred_folder, file)
        gt_path   = os.path.join(gt_folder, file)

        if not os.path.exists(gt_path):
            print("Missing GT:", file)
            continue

        pred = nib.load(pred_path).get_fdata().astype(int)
        gt   = nib.load(gt_path).get_fdata().astype(int)

        n_images += 1

        for c in classes:

            TP, FP, FN = structure_metrics(pred, gt, c, iou_threshold)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            iou50     = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

            per_image_metrics[c]["precision"].append(precision)
            per_image_metrics[c]["recall"].append(recall)
            per_image_metrics[c]["f1"].append(f1)
            per_image_metrics[c]["iou50"].append(iou50)


    # -----------------------------
    # Final results
    # -----------------------------
    print(f"\nEvaluated on {n_images} images\n")
    print("Structure-based metrics with 95% CI\n")

    mean_precisions = []
    mean_recalls = []
    mean_f1s = []

    for c in classes:

        p_mean, p_ci = compute_ci(per_image_metrics[c]["precision"])
        r_mean, r_ci = compute_ci(per_image_metrics[c]["recall"])
        f1_mean, f1_ci = compute_ci(per_image_metrics[c]["f1"])
        iou_mean, iou_ci = compute_ci(per_image_metrics[c]["iou50"])

        mean_precisions.append(p_mean)
        mean_recalls.append(r_mean)
        mean_f1s.append(f1_mean)

        print(f"Class {c}")
        print(f" F1:        {f1_mean:.3f} ± {f1_ci:.3f}")
        print(f" Precision: {p_mean:.3f} ± {p_ci:.3f}")
        print(f" Recall:    {r_mean:.3f} ± {r_ci:.3f}")
        print(f" IoU@50:    {iou_mean:.3f} ± {iou_ci:.3f}")
        print(
    f"& {f1_mean:.3f} $\\pm$ {f1_ci:.3f} "
    f"& {p_mean:.3f} $\\pm$ {p_ci:.3f} "
    f"& {r_mean:.3f} $\\pm$ {r_ci:.3f} "
    f"& {iou_mean:.3f} $\\pm$ {iou_ci:.3f} "
)

    print("Mean over classes")
    print(f" Precision: {np.mean(mean_precisions):.3f}")
    print(f" Recall:    {np.mean(mean_recalls):.3f}")
    print(f" F1:        {np.mean(mean_f1s):.3f}")




def mask_nms(preds, iou_threshold=0.5):
    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    kept = []
    for conf, mask in preds:
        keep = True
        for kept_conf, kept_mask in kept:
            if compute_iou(mask, kept_mask) > iou_threshold:
                keep = False
                break
        if keep:
            kept.append((conf, mask))
    return kept

def compute_ap(prob_map, gt_mask, class_id, iou_threshold):
    prob = prob_map[class_id, 0, :, :]
    gt_bin = (gt_mask == class_id)
    gt_labels, n_gt = label(gt_bin)
    if n_gt == 0:
        return None
    gt_objects = [gt_labels == g for g in range(1, n_gt+1)]
    thresholds = np.linspace(0.05, 0.95, 10)
    preds = []

    for t in thresholds:
        pred_mask = prob > t
        pred_labels, n_pred = label(pred_mask)
        for p in range(1, n_pred+1):
            mask = pred_labels == p
            if mask.sum() < 50:
                continue
            confidence = prob[mask].max()           #option to do mean() here
            preds.append((confidence, mask))

    preds = mask_nms(preds, iou_threshold=0.5)

    if len(preds) == 0:
        return 0.0
    matched_gt = np.zeros(n_gt, dtype=bool)

    TP = []
    FP = []

    for conf, pred_mask in preds:
        best_iou = 0
        best_gt = -1
        for i, gt_mask_i in enumerate(gt_objects):
            if matched_gt[i]:
                continue
            iou = compute_iou(pred_mask, gt_mask_i)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= iou_threshold:
            TP.append(1)
            FP.append(0)
            matched_gt[best_gt] = True
        else:
            TP.append(0)
            FP.append(1)

    TP = np.array(TP)
    FP = np.array(FP)
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)

    precision = cum_TP / (cum_TP + cum_FP + 1e-8)
    recall = cum_TP / n_gt
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap



def compute_coco_ap(prob_map, gt_mask, class_id, iou_thresholds):

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for t in iou_thresholds:
        ap = compute_ap(prob_map, gt_mask, class_id, t)
        aps.append(ap)

    return np.mean(aps)




def calcuate_APa50(pred_folder, gt_folder, classes, iou_thresholds):
    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    ap_per_class = {c: [] for c in classes}

    for file in os.listdir(pred_folder):

        if not file.endswith(".npz"):
            continue

        case_name = file.replace(".npz", "")

        pred_path = os.path.join(pred_folder, file)
        gt_path = os.path.join(gt_folder, case_name + ".nii.gz")

        if not os.path.exists(gt_path):
            continue

        data = np.load(pred_path)
        prob_map = data['probabilities']

        gt_mask = nib.load(gt_path).get_fdata().astype(int)

        for c in classes:
            ap = compute_coco_ap(prob_map, gt_mask, c, iou_thresholds)
            ap_per_class[c].append(ap)

    # -----------------------------
    # FINAL mAP@50
    # -----------------------------
    print("\nResults:\n")

    mean_aps = []

    for c in classes:
        mean_ap = np.mean(ap_per_class[c])
        mean_aps.append(mean_ap)

        print(f"Class {c} AP@[.50:.95]: {mean_ap:.4f}")

    mAP = np.mean(mean_aps)

    print(f"\n mAP@[.50:.95]: {mAP:.4f}")


# I ADDED HER FOR MACRO
def macro_metrics_skip_empty(pred_folder, gt_folder, classes, list_of_interest=None):
    # store all metrics
    per_class_metrics = {
        c: {"dice": [], "iou": [], "precision": [], "recall": []}
        for c in classes
    }

    n_images = 0

    # -----------------------------
    # Loop over dataset
    # -----------------------------
    for file in os.listdir(pred_folder):

        if not file.endswith(".nii.gz"):
            continue

        case_name = file.replace(".nii.gz", "")
        if list_of_interest is not None and case_name not in list_of_interest:
            continue

        pred_path = os.path.join(pred_folder, file)
        gt_path   = os.path.join(gt_folder, file)

        if not os.path.exists(gt_path):
            continue

        pred = nib.load(pred_path).get_fdata().astype(int)
        gt   = nib.load(gt_path).get_fdata().astype(int)

        n_images += 1

        for c in classes:

            pred_bin = (pred == c)
            gt_bin   = (gt == c)

            TP = np.logical_and(pred_bin, gt_bin).sum()
            FP = np.logical_and(pred_bin, ~gt_bin).sum()
            FN = np.logical_and(~pred_bin, gt_bin).sum()

            denom_dice = 2 * TP + FP + FN

            # 🔑 skip empty cases (same logic for ALL metrics)
            if denom_dice == 0:
                continue

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
            iou       = TP / (TP + FP + FN)
            dice      = (2 * TP) / denom_dice

            per_class_metrics[c]["precision"].append(precision)
            per_class_metrics[c]["recall"].append(recall)
            per_class_metrics[c]["iou"].append(iou)
            per_class_metrics[c]["dice"].append(dice)

    # -----------------------------
    # Final aggregation
    # -----------------------------
    print(f"\nEvaluated on {n_images} images\n")

    macro_results = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": []
    }

    for c in classes:

        values = per_class_metrics[c]

        if len(values["dice"]) == 0:
            print(f"Class {c}: No valid samples")
            continue

        d_mean = np.mean(values["dice"])
        i_mean = np.mean(values["iou"])
        p_mean = np.mean(values["precision"])
        r_mean = np.mean(values["recall"])

        macro_results["dice"].append(d_mean)
        macro_results["iou"].append(i_mean)
        macro_results["precision"].append(p_mean)
        macro_results["recall"].append(r_mean)

        print(f"Class {c} (N={len(values['dice'])})")
        print(f" Dice:      {d_mean:.4f}")
        print(f" IoU:       {i_mean:.4f}")
        print(f" Precision: {p_mean:.4f}")
        print(f" Recall:    {r_mean:.4f}")
        print()

    # -----------------------------
    # Macro over classes
    # -----------------------------
    if len(macro_results["dice"]) > 0:
        print("Macro over classes:")
        print(f" Dice:      {np.mean(macro_results['dice']):.4f}")
        print(f" IoU:       {np.mean(macro_results['iou']):.4f}")
        print(f" Precision: {np.mean(macro_results['precision']):.4f}")
        print(f" Recall:    {np.mean(macro_results['recall']):.4f}")

    return per_class_metrics