import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from nnunetv2.inference.predict_from_raw_data import predict_entry_point

from nnunetv2.evaluation.evaluate_predictions import evaluate_folder_entry_point


def custom_prediction_and_evaluation(list_chk, tf_with_tta, type_mask, trainer, planner_json_name, config, json_output_folder, pof, task_name, task_id, folds, is_imagesTs=True, save_prob = True):
    print("Starting...")
    if is_imagesTs:
        images_folder_name = "imagesTs"
        labels_folder_name = "labelsTs"
        path_add = ""
    else:
        images_folder_name = "imagesTr"
        labels_folder_name = "labelsTr"
        path_add = "_of_imagesTr"
    for chk in list_chk:
        for tmask in type_mask:        
            intermediate_file = f"{trainer}__{planner_json_name}__{config}"
            prediction_input_folder = os.path.join(json_output_folder, images_folder_name)
            if is_imagesTs:
                path_labelsTs = os.path.join(os.environ["nnUNet_raw"], task_name, labels_folder_name, tmask)
            else:
                path_labelsTs = os.path.join(os.environ["nnUNet_raw"], task_name, labels_folder_name)
            if tf_with_tta:
                prediction_output_folder = os.path.join(pof, task_name, intermediate_file, tmask, chk +"_with_tta"+path_add)
                evaluate_out_path = os.path.join(pof, task_name,intermediate_file, "summary_"+tmask+"_"+ chk +"_with_tta"+path_add+".json")
            else:
                prediction_output_folder = os.path.join(pof, task_name, intermediate_file, tmask, chk +path_add)  
                evaluate_out_path = os.path.join(pof, task_name,intermediate_file, "summary_"+tmask+"_"+ chk +path_add+".json")

            path_djfile = os.path.join(prediction_output_folder,"dataset.json")
            path_pfile = os.path.join(prediction_output_folder,"plans.json")


            sys.argv = [
            "nnUNetv2_predict",
            "-i", prediction_input_folder,
            "-o", prediction_output_folder,
            "-d", str(task_id),
            "-c", config,
            "-f", folds,
            "-tr", trainer,                    # only if custom trainer
            "-p", planner_json_name,           # only if custom planner
            "-device", "cpu",
            "-chk", chk+".pth",
            "--disable_progress_bar"
            ]
            if not tf_with_tta:
                sys.argv.append("--disable_tta")
            if save_prob:
                sys.argv.append("--save_probabilities")

            predict_entry_point()

            sys.argv = [
                "nnUNetv2_evaluate_folder",
                "-djfile", path_djfile,
                "-pfile", path_pfile,
                path_labelsTs,
                prediction_output_folder,
                "-o", evaluate_out_path
                ]

            evaluate_folder_entry_point()

            print(f"File ={chk} with tta =={tf_with_tta} regarding the {tmask} masks Saved.")

    print("Finished.")



def display_prediction_graphs(chk_list, value_of_pred_in_mask, type_mask, tf_with_tta ,pof, task_name, trainer, planner_json_name, config):
    
    intermediate_file = f"{trainer}__{planner_json_name}__{config}"

    # Lists to store metrics tf_with_tta
    mean_dice_list_spoca = []
    mean_iou_list_spoca = []
    FP_list_spoca = []
    FN_list_spoca = []
    TP_list_spoca = []
    TN_list_spoca = []
    mean_dice_list_spoca_with_tta = []
    mean_iou_list_spoca_with_tta = []
    FP_list_spoca_with_tta = []
    FN_list_spoca_with_tta = []
    TP_list_spoca_with_tta = []
    TN_list_spoca_with_tta = []

    mean_dice_list_region = []
    mean_iou_list_region = []
    FP_list_region = []
    FN_list_region = []
    TP_list_region = []
    TN_list_region = []
    mean_dice_list_region_with_tta = []
    mean_iou_list_region_with_tta = []
    FP_list_region_with_tta = []
    FN_list_region_with_tta = []
    TP_list_region_with_tta = []
    TN_list_region_with_tta = []

    for c in chk_list:
        for d in type_mask:
            if not tf_with_tta:
                path_summary = os.path.join(pof, task_name, intermediate_file, f"summary_{d}_checkpoint_epoch_{c}.json")    
                with open(path_summary, "r") as f:
                    summary = json.load(f)
                mean_metrics = summary['mean'][value_of_pred_in_mask]  # usually 'mean' -> class -> metrics
            if tf_with_tta:
                path_summary_with_tta = os.path.join(pof, task_name, intermediate_file, f"summary_{d}_checkpoint_epoch_{c}_with_tta.json")
                with open(path_summary_with_tta, "r") as f:
                    summary_with_tta = json.load(f)
                mean_metrics_with_tta = summary_with_tta['mean'][value_of_pred_in_mask]
            
            if d == "spoca":
                if not tf_with_tta:
                    mean_dice_list_spoca.append(mean_metrics['Dice'])
                    mean_iou_list_spoca.append(mean_metrics['IoU'])
                    FP_list_spoca.append(mean_metrics['FP'])
                    FN_list_spoca.append(mean_metrics['FN'])
                    TP_list_spoca.append(mean_metrics['TP'])
                    TN_list_spoca.append(mean_metrics['TN'])
                if tf_with_tta:
                    mean_dice_list_spoca_with_tta.append(mean_metrics_with_tta['Dice'])
                    mean_iou_list_spoca_with_tta.append(mean_metrics_with_tta['IoU'])
                    FP_list_spoca_with_tta.append(mean_metrics_with_tta['FP'])
                    FN_list_spoca_with_tta.append(mean_metrics_with_tta['FN'])
                    TP_list_spoca_with_tta.append(mean_metrics_with_tta['TP'])
                    TN_list_spoca_with_tta.append(mean_metrics_with_tta['TN'])

            elif d == "region":
                if not tf_with_tta:
                    mean_dice_list_region.append(mean_metrics['Dice'])
                    mean_iou_list_region.append(mean_metrics['IoU'])
                    FP_list_region.append(mean_metrics['FP'])
                    FN_list_region.append(mean_metrics['FN'])
                    TP_list_region.append(mean_metrics['TP'])
                    TN_list_region.append(mean_metrics['TN'])
                if tf_with_tta:
                    mean_dice_list_region_with_tta.append(mean_metrics_with_tta['Dice'])
                    mean_iou_list_region_with_tta.append(mean_metrics_with_tta['IoU'])
                    FP_list_region_with_tta.append(mean_metrics_with_tta['FP'])
                    FN_list_region_with_tta.append(mean_metrics_with_tta['FN'])
                    TP_list_region_with_tta.append(mean_metrics_with_tta['TP'])
                    TN_list_region_with_tta.append(mean_metrics_with_tta['TN'])


    # --- Plot Mean Dice and IOU ---
    plt.figure(figsize=(8,5))
    if "spoca" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, mean_dice_list_spoca, marker='o', color = "blue", label='Mean Dice in spoca masks')
            plt.plot(chk_list, mean_iou_list_spoca, marker='s', color = "blue", label='Mean IoU in spoca masks')
        if tf_with_tta:
            plt.plot(chk_list, mean_dice_list_spoca_with_tta, marker='o', color = "lightblue", label='Mean Dice in spoca masks with tta')
            plt.plot(chk_list, mean_iou_list_spoca_with_tta, marker='s', color = "lightblue", label='Mean IoU in spoca masks with tta')
            
    if "region" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, mean_dice_list_region, marker='o', color = "red", label='Mean Dice in region masks')
            plt.plot(chk_list, mean_iou_list_region, marker='s', color = "red", label='Mean IoU in region masks')
        if tf_with_tta:
            plt.plot(chk_list, mean_dice_list_region_with_tta, marker='o', color = "coral", label='Mean Dice in region masks with tta')
            plt.plot(chk_list, mean_iou_list_region_with_tta, marker='s', color = "coral", label='Mean IoU in region masks with tta')
            
    plt.xlabel("Checkpoint")
    plt.ylabel("Metric Value")
    plt.title("Mean Dice and IoU of spoca and region masks over Checkpoints")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot FN and FP ---
    plt.figure(figsize=(8,5))
    if "spoca" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, FN_list_spoca, marker='o', color = "blue", label='mean False Negatives in spoca masks (FN)')
            plt.plot(chk_list, FP_list_spoca, marker='s', color = "blue", label='mean False Positives in spoca masks (FP)')
        if tf_with_tta:
            plt.plot(chk_list, FN_list_spoca_with_tta, marker='o', color = "lightblue", label='mean False Negatives in spoca masks with tta (FN)')
            plt.plot(chk_list, FP_list_spoca_with_tta, marker='s', color = "lightblue", label='mean False Positives in spoca masks with tta (FP)')
    if "region" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, FN_list_region, marker='o', color = "red", label='mean False Negatives in region masks (FN)')
            plt.plot(chk_list, FP_list_region, marker='s', color = "red", label='mean False Positives in region masks (FP)')
        if tf_with_tta:
            plt.plot(chk_list, FN_list_region_with_tta, marker='o', color = "coral", label='mean False Negatives in region masks with tta (FN)')
            plt.plot(chk_list, FP_list_region_with_tta, marker='s', color = "coral", label='mean False Positives in region masks with tta (FP)')
    plt.xlabel("Checkpoint")
    plt.ylabel("Count")
    plt.title("FN and FP of spoca and region masks over Checkpoints")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot TN and TP ---
    plt.figure(figsize=(8,5))
    if "spoca" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, TN_list_spoca, marker='o', color = "blue", label='mean True Negatives in spoca masks (TN)')
        if tf_with_tta:
            plt.plot(chk_list, TN_list_spoca_with_tta, marker='o', color = "lightblue", label='mean True Negatives in spoca masks with tta (TN)')
    if "region" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, TN_list_region, marker='o', color = "red", label='mean True Negatives in region masks (TN)')
        if tf_with_tta:
            plt.plot(chk_list, TN_list_region_with_tta, marker='o', color = "coral", label='mean True Negatives in region masks with tta (TN)')
    plt.xlabel("Checkpoint")
    plt.ylabel("Count")
    plt.title("TN and TP of spoca and region masks over Checkpoints")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    if "spoca" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, TP_list_spoca, marker='s', color = "blue", label='mean True Positives in spoca masks (TP)')
        if tf_with_tta:
            plt.plot(chk_list, TP_list_spoca_with_tta, marker='s', color = "lightblue", label='mean True Positives in spoca masks with tta(TP)')
    if "region" in type_mask:
        if not tf_with_tta:
            plt.plot(chk_list, TP_list_region, marker='s', color = "red", label='mean True Positives in region masks (TP)')
        if tf_with_tta:
            plt.plot(chk_list, TP_list_region_with_tta, marker='s', color = "coral", label='mean True Positives in region masks with tta (TP)')
    plt.xlabel("Checkpoint")
    plt.ylabel("Count")
    plt.title("TN and TP of spoca and region masks over Checkpoints")
    plt.legend()
    plt.grid(True)
    plt.show()
