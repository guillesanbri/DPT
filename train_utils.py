import tqdm
import torch
import wandb
import numpy as np
from torch.cuda.amp import autocast


def train(dataloader, model, loss_fn, optimizer, training_step, scaler, accumulation_steps, mixed_precision, device):
    size = len(dataloader.dataset)
    model.train()
    model.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        training_step += 1
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        with autocast(enabled=mixed_precision):
            pred = model(X)
            masked_pred, masked_y = mask_predictions(pred, y)
            loss = loss_fn(masked_pred, masked_y)
            loss = loss / accumulation_steps

        # Backpropagation
        scaler.scale(loss).backward()
        if (batch+1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

        if batch % 500 == 0:
            loss, current = loss.item() * accumulation_steps, batch * len(X)
            metric_names = ["train_silog", "train_log10", "train_abs_rel", "train_sq_rel",
                            "train_rmse", "train_rmse_log", "train_d1", "train_d2", "train_d3"]
            metrics = np.array(compute_errors(masked_y.cpu().detach().numpy(), masked_pred.cpu().detach().numpy()))
            wandb.log({"training_step": training_step, "train_loss": loss, **dict(zip(metric_names, metrics))})
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if np.isnan(loss):  # Stop run if loss is nan
                return -1
        # if batch % 1000 == 0 and batch != 0:
        #      return training_step  # premature exit
    return training_step


def test(dataloader, model, loss_fn, training_step, device, log_wandb=True):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    metrics = np.array([0.0 for _ in range(9)])
    metric_names = ["val_silog", "val_log10", "val_abs_rel", "val_sq_rel",
                    "val_rmse", "val_rmse_log", "val_d1", "val_d2", "val_d3"]
    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            masked_pred, masked_y = mask_predictions(pred, y)
            loss += loss_fn(masked_pred, masked_y).item()
            metrics += np.array(compute_errors(masked_y.cpu().numpy(), masked_pred.cpu().numpy()))
    loss /= num_batches
    metrics /= num_batches
    if log_wandb:
        wandb.log({"training_step": training_step, "val_loss": loss, **dict(zip(metric_names, metrics))})
    metrics_string = " | ".join([f"{metric_name}: {metric:.3f}" for metric_name, metric in zip(metric_names, metrics)])
    print(f"Validation metrics (Avg): loss: {loss:.6f} | " + metrics_string + "\n")


# From eval_with_pngs.py
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3]


def mask_predictions(output, target):
    output_1 = output / 255
    target_1 = target[..., 0]
    min_depth_eval = 1e-3 / 255
    max_depth_eval = 80 / 255
    # Trim prediction values (!) Inplace operation? TODO
    output_1[output_1 < min_depth_eval] = min_depth_eval
    output_1[output_1 > max_depth_eval] = max_depth_eval
    output_1[torch.isinf(output_1)] = max_depth_eval
    # Trim ground truth
    target_1[torch.isinf(target_1)] = 0
    target_1[torch.isnan(target_1)] = 0
    # Generate mask where target values are not zero and not inf
    valid_mask = torch.logical_and(target_1 > min_depth_eval, target_1 < max_depth_eval)
    masked_target = target_1[valid_mask]
    masked_output = output_1[valid_mask]
    return masked_output, masked_target


# adapted from https://github.com/imran3180/depth-map-prediction/blob/master/main.py
# and eval_with_pngs to match eq. 4 from https://arxiv.org/pdf/1406.2283.pdf
def custom_loss(masked_output, masked_target):
    # output_cv2 = output.detach().cpu().numpy()[0].astype(np.uint8)
    # target_cv2 = (target.detach().cpu().numpy()[0]*255).astype(np.uint8)
    # cv2.imshow("output", output_cv2)
    # cv2.imshow("target", target_cv2)
    # cv2.waitKey(0)

    di = torch.log(masked_output) - torch.log(masked_target)
    n = masked_output.shape[0]
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2) / n
    second_term = 0.5 * torch.pow(torch.sum(di), 2) / (n ** 2)  # TODO: 0.5 is lambda, which could be tuned
    loss = fisrt_term - second_term
    return loss.mean()