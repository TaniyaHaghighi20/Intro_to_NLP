from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, classification_report, auc
import torch
from models.focal_loss import FocalLoss
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score, MulticlassAUROC, MulticlassPrecisionRecallCurve
from torchmetrics.classification.accuracy import MulticlassAccuracy
import math
from tqdm import tqdm

def macro_ap_(ap, classes):
    precision, recall, _ = ap
    indices = torch.tensor(list(sorted(classes.values())))
    # Calculating area under the curve for each class
    aps = []
    for i in indices:
        # Sort the recall and precision arrays
        recall_sorted_indices = recall[i].argsort()
        sorted_recall = recall[i][recall_sorted_indices]
        sorted_precision = precision[i][recall_sorted_indices]

        # Calculate AUC using sklearn's auc function
        class_auc = auc(sorted_recall.cpu().numpy(), sorted_precision.cpu().numpy())
        if math.isnan(class_auc):
            class_auc = 0.0
        aps.append(class_auc)
    return aps
     
def evaluate(model, vae, dataloader, device, num_classes, dataset_name, return_raw=False, show_progress=False):
    """
    Evaluate the model on a given dataloader. Returns:
      - avg_loss, auc, ap (average precision),
        f1 (macro-average), acc (accuracy).

    If return_raw=True, also returns (all_labels, all_preds) for per-class metrics.
    If show_progress=True, we wrap the dataloader in a tqdm progress bar.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_losses = []
    criterion = FocalLoss()


    auc = MulticlassAUROC(num_classes=num_classes, average=None, sync_on_compute=False).to(device)
    ap = MulticlassPrecisionRecallCurve( num_classes=num_classes, sync_on_compute=False).to(device)
    f1 = MulticlassF1Score(num_classes=num_classes, average=None, sync_on_compute=False).to(device)
    acc = MulticlassAccuracy(num_classes=num_classes, average=None, sync_on_compute=False).to(device)
    pr = MulticlassPrecision(num_classes=num_classes, average=None, sync_on_compute=False).to(device)

    # Optionally wrap the dataloader in a tqdm progress bar
    loader = dataloader
    if show_progress:
        loader = tqdm(dataloader, desc=f"Evaluating on {dataset_name}", leave=False)

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.long().to(device)

            # Encode images with VAE
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # Use a dummy t=0 or t=1 (any valid t range)
            t = torch.randint(0, 1, (x.shape[0],), device=device)
            # Use the last class index as the null class (for CFG)
            y_null = torch.tensor([num_classes - 1] * y.shape[0], device=device)

            out = model(x, t, y_null)  # (N, num_classes)
            loss = criterion(out, y).mean()
            all_losses.append(loss.item())
            
            auc.update(out, y)
            ap.update(out, y)
            f1.update(out, y)
            acc.update(out, y)
            pr.update(out, y)

            # Convert model outputs to probabilities
            preds = F.softmax(out, dim=1).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()

            all_labels.append(labels)
            all_preds.append(preds)

           
            

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    avg_loss = np.mean(all_losses)

    if return_raw:
        return avg_loss, auc.compute(), ap.compute(), f1.compute(), acc.compute(), pr.compute(), all_labels, all_preds
    else:
        return avg_loss, auc.compute(), ap.compute(), f1.compute(), acc.compute(), pr.compute()


def cross_evaluate(model, vae, source_dm, all_data_modules, device, rank, args, experiment_dir):
    """
    Evaluate 'model' (trained on source_dm) on the test sets (filtered) of *all* data modules.
    Log metrics (including per-class) and compute an overall average.
    """
    # Only rank=0 will write to file
    if rank != 0:
        return

    results_file = os.path.join(experiment_dir, "cross_eval_results.txt")
    done_marker = f"[DONE] {source_dm.dataset_name}"

    # If the file exists and has the done marker, skip
    if os.path.exists(results_file):
        with open(results_file, "r") as ff:
            lines = ff.read()
            if done_marker in lines:
                print(f"[cross_evaluate] Skipping cross-eval for {source_dm.dataset_name}: DONE marker found.")
                return  # skip entirely

    existing_lines = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            existing_lines = f.read().splitlines()

    # Lists to store metrics across all test modules (to compute an average at the end)
    all_aucs = []
    all_f1s  = []
    all_aps  = []
    all_accs = []
    all_prs = []

    with open(results_file, "a") as f:
        f.write(f"\n\n======================================\n")
        f.write(f"  Model trained on: {source_dm.dataset_name}\n")
        f.write(f"======================================\n")

        for target_dm in all_data_modules:
            # If skipping, check if the dataset_name is already in file
            if any(target_dm.dataset_name in line for line in existing_lines):
                print(f"[cross_evaluate] Skipping {target_dm.dataset_name}, already in cross_eval_results.txt")
                continue

            if target_dm.dataset_name != source_dm.dataset_name and (target_dm.dataset_name in ["DS7", "DS8"] or source_dm.dataset_name in ["DS7", "DS8"]):
                continue
            # Build a filtered test set for the target data module
            filtered_testset = target_dm.filtered_testset(source_dm.classes)
            print("source_dm:",source_dm.classes)
            print("target_dm:",target_dm.classes)
            print("filtered_classes:",target_dm.filtered_classes)
            indices_ = list(target_dm.filtered_classes.values())
            indices = torch.tensor(indices_, device=device)

            test_sampler = DistributedSampler(
                filtered_testset,
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=False,
                seed=args.global_seed
            )
            test_loader = DataLoader(
                filtered_testset,
                batch_size=int(args.global_batch_size // dist.get_world_size()),
                sampler=test_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            # Evaluate -> return raw predictions for per-class metrics
            avg_loss, auc_, ap_, f1_, acc_, pr_, all_labels, all_preds = evaluate(
                model, vae, test_loader, device,
                num_classes=len(source_dm.classes),
                return_raw=True,
                show_progress=True,  # turn on the batch-level progress
                dataset_name=target_dm.dataset_name
            )
            ap_ = macro_ap_(ap_, target_dm.filtered_classes)
             # Calculate macro averages
            macro_auc = auc_[indices].mean().item()
            macro_ap =  sum(ap_)/len(ap_)
            macro_f1 = f1_[indices].mean().item()
            macro_acc = acc_[indices].mean().item()
            macro_precision = pr_[indices].mean().item()

            # Log per-class metrics
            f.write(f"\n================= {target_dm.dataset_name} =================\n")
            f.write(f"Macro Metrics:\n")
            f.write(f"  AUC:       {macro_auc:.4f}\n")
            f.write(f"  AP:        {macro_ap:.4f}\n")
            f.write(f"  F1:        {macro_f1:.4f}\n")
            f.write(f"  Accuracy:  {macro_acc:.4f}\n")
            f.write(f"  Precision: {macro_precision:.4f}\n")
            f.write(f"Loss:      {avg_loss:.4f}\n")
            f.write("Per-Class Metrics:\n")

            # Store for averaging
            all_aucs.append(auc_[indices])
            all_f1s.append(f1_[indices])
            all_aps.append(ap_)
            all_accs.append(acc_[indices])
            all_prs.append(pr_[indices])

            for idx, class_name in enumerate(target_dm.filtered_classes):
                f.write(f"  Class {class_name}:\n")
                f.write(f"    AUC:       {auc_[idx]:.4f}\n")
                f.write(f"    AP:        {ap_[idx]:.4f}\n")
                f.write(f"    F1:        {f1_[idx]:.4f}\n")
                f.write(f"    Accuracy:  {acc_[idx]:.4f}\n")
                f.write(f"    Precision: {pr_[idx]:.4f}\n")

        # Compute overall average across all test data modules
        avg_auc = torch.cat(all_aucs).mean(dim=0).item() if len(all_aucs) > 0 else 0
        avg_ap  = np.mean(np.concatenate(all_aps))  if len(all_aps) > 0 else 0
        avg_f1  = torch.cat(all_f1s).mean(dim=0).item()  if len(all_f1s) > 0 else 0
        avg_acc = torch.cat(all_accs).mean(dim=0).item() if len(all_accs) > 0 else 0
        avg_pr = torch.cat(all_prs).mean(dim=0).item() if len(all_prs) > 0 else 0

        f.write("\n===== Average Across All Test Sets =====\n")
        f.write(f"AUC (avg):       {avg_auc:.4f}\n")
        f.write(f"AP (avg):        {avg_ap:.4f}\n")
        f.write(f"F1 (avg):        {avg_f1:.4f}\n")
        f.write(f"Accuracy (avg):  {avg_acc:.4f}\n")
        f.write(f"Precision (avg):  {avg_pr:.4f}\n")
        f.write("=========================================\n\n")
        f.write(done_marker + "\n")

