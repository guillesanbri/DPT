import os
import cv2
import time
import tqdm
import wandb
import torch
import numpy as np

from KITTIDataset import KITTIDataset
from torch.utils.data import DataLoader
from util.misc import get_random_string
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

# Hyperparameters and config
# net_w, net_h = 1216, 352  # TODO: Try full size and/or bigger bs with more VRAM
# TODO: Add horizontal flip augmentation
net_w = 640
net_h = 192
batch_size = 1
accumulation_steps = 1
epochs = 2
learning_rate = 1e-5
# memory_compressed only supports batch_size=1
backbone = "vitb_effb0"  # "vitb_rn50_384"
backbone = "vitb_rn50_384"
attention_variant = None  # "performer"
attention_heads = 8
hooks = [0, 1, 8, 11]
hooks = "str:0,1,8,11"
train_images_file = "train_files_eigen_full_fix.txt"
val_images_file = "val_files_eigen_full_fix.txt"
output_name = "dpt_hybrid_custom-kitti-" + get_random_string(8)
output_filename = output_name + ".pt"
opt = torch.optim.AdamW

config_dict = {
    "backbone": backbone,
    "attention_variant": attention_variant,
    "memory_compressed_rate": 2,
    "attention_heads": attention_heads,
    "hooks": hooks,
    "mixed_precision": True,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "accumulation_steps": accumulation_steps,
    "optimizer": opt,
    "input_height": net_h,
    "input_width": net_w,
    "downsampling": "Resize image along w and h",  # TODO: Test resize min axis to 384 and random crop.
    "train_images_file": train_images_file,
    "val_images_file": val_images_file,
    "weights_file_name": output_filename
}


def train(dataloader, model, loss_fn, optimizer, training_step, scaler):
    size = len(dataloader.dataset)
    model.train()
    model.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        training_step += 1
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        with autocast():
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
        # if batch % 1000 == 0 and batch != 0:
        #      return training_step  # premature exit
    return training_step


def test(dataloader, model, loss_fn, training_step, log_wandb=True):
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


if __name__ == "__main__":
    # Init wandb
    wandb.init(project="efficientnet", config=config_dict)  # DPT
    config = wandb.config
    accumulation_steps = config["accumulation_steps"]
    learning_rate = config["learning_rate"]
    attention_heads = config["attention_heads"]
    hooks = config["hooks"]

    if isinstance(hooks, str):
        hooks = [int(hook) for hook in hooks[4:].split(",")]

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Load model to selected device
    model = DPTDepthModel(
                path="weights/dpt_hybrid-midas-d889a10e.pt",
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone=backbone,
                attention_heads=attention_heads,
                hooks=hooks,
                non_negative=True,
                enable_attention_hooks=False,
                attention_variant=attention_variant).to(device)

    # Transformations
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
             Resize(
                net_w,
                net_h,
                # resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
             ),
             normalization,
             PrepareForNet(),
        ]
    )

    # Create datasets
    dataset_dir = "../data/KITTI/"
    image_dir = os.path.join(dataset_dir, "raw")
    depth_dir = os.path.join(dataset_dir, "data_depth_annotated/")
    train_images_file_path = os.path.join(dataset_dir, train_images_file)
    val_filenames_file = os.path.join(dataset_dir, val_images_file)
    train_dataset = KITTIDataset(image_dir, depth_dir, train_images_file_path, transform=transform)
    val_dataset = KITTIDataset(image_dir, depth_dir, val_filenames_file, transform=transform)
    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # Print shape info
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    loss_fn = custom_loss
    optimizer = opt(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    training_step = 0
    # Train loop
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    t0 = time.time()
    # test(test_dataloader, model, loss_fn, training_step, log_wandb=False)
    # wandb.log({"validation_inference_time": time.time()-t0})
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_step = train(train_dataloader, model, loss_fn, optimizer, training_step, scaler)
        # print(f"1000 batches ejecutados en: {time.time()-t0}")
        # exit()
        test(test_dataloader, model, loss_fn, training_step)
        checkpoint_filename = "weights/" + output_name + "_" + str(t+1).zfill(3) + ".pt"
        torch.save(model.state_dict(), checkpoint_filename)
        print(f"Saved PyTorch Model checkpoint to {checkpoint_filename}")
    print("Done!")

    torch.save(model.state_dict(), "weights/" + output_filename)
    print(f"Saved PyTorch Model State to weights/{output_filename}")
