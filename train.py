import os
import cv2
import time
import wandb
import torch

from KITTIDataset import KITTIDataset
from torch.utils.data import DataLoader
from util.misc import get_random_string
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet, RandomHorizontalFlip
from train_utils import custom_loss, train, test

from torch.cuda.amp import GradScaler

# Hyperparameters and config
# Input
net_w, net_h = 640, 192
# Training
base_path = "weights/dpt_hybrid-midas-d889a10e.pt"
mixed_precision = False
batch_size, accumulation_steps = 1, 1
epochs = 20
learning_rate = 1e-5
opt = torch.optim.AdamW
train_images_file = "train_files_eigen_full_fix.txt"
val_images_file = "val_files_eigen_full_fix.txt"
# Model architecture
backbone = "vitb_rn50_384"  # "vitb_effb0"
transformer_hooks = "str:8,11"
attention_variant = None  # "performer"
attention_heads = 8
# Output
output_name = "dpt_hybrid_custom-kitti-" + get_random_string(8)
output_filename = output_name + ".pt"

config_dict = {
    "input_height": net_h,
    "input_width": net_w,
    "downsampling": "Resize image along w and h",

    "base_path": base_path,
    "mixed_precision": mixed_precision,
    "batch_size": batch_size,
    "accumulation_steps": accumulation_steps,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "optimizer": opt,
    "train_images_file": train_images_file,
    "val_images_file": val_images_file,

    "backbone": backbone,
    "transformer_hooks": transformer_hooks,
    "attention_variant": attention_variant,
    "attention_heads": attention_heads,

    "weights_file_name": output_filename
}


if __name__ == "__main__":
    # Init wandb
    wandb.init(project="efficientnet", config=config_dict)  # DPT
    config = wandb.config
    # Re-read config for wandb-sweep-managed training
    learning_rate = config["learning_rate"]
    accumulation_steps = config["accumulation_steps"]
    backbone = config["backbone"]
    transformer_hooks = config["transformer_hooks"]
    attention_variant = config["attention_variant"]
    if attention_variant == "None":
        attention_variant = None
    attention_heads = config["attention_heads"]

    # Convert str hooks to list (wandb hacky solution to display hooks correctly)
    assert isinstance(transformer_hooks, str) and transformer_hooks[:4] == "str:", \
        'Hooks are not in the format "str:[att_hook1, att_hook2]"'
    conv_hooks = {"vitb_rn50_384": [0, 1], "vitb_effb0": [1, 2]}[backbone]
    transformer_hooks = [int(hook) for hook in transformer_hooks[4:].split(",")]
    hooks = conv_hooks + transformer_hooks

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Load model to selected device
    model = DPTDepthModel(
                path=base_path,
                scale=0.00006016,  # KITTI
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
    transform_train = Compose(
        [
             Resize(net_w, net_h, keep_aspect_ratio=True, ensure_multiple_of=32,
                    resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
             RandomHorizontalFlip(0.5),
             normalization,
             PrepareForNet(),
        ]
    )
    transform_val = Compose(
        [
             Resize(net_w, net_h, keep_aspect_ratio=True, ensure_multiple_of=32,
                    resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
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
    train_dataset = KITTIDataset(image_dir, depth_dir, train_images_file_path, transform=transform_train)
    val_dataset = KITTIDataset(image_dir, depth_dir, val_filenames_file, transform=transform_val)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # Print shape info
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    loss_fn = custom_loss
    optimizer = opt(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=mixed_precision)
    training_step = 0

    # Train loop
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # t0 = time.time()
    # test(test_dataloader, model, loss_fn, training_step, log_wandb=False)
    # wandb.log({"validation_inference_time": time.time()-t0})
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_step = train(train_dataloader, model, loss_fn, optimizer, training_step, scaler, accumulation_steps, mixed_precision, device)
        if training_step < 0:
            break
        # print(f"1000 batches ejecutados en: {time.time()-t0}")
        # exit()
        test(test_dataloader, model, loss_fn, training_step, device)
        checkpoint_filename = "weights/" + output_name + "_" + str(t+1).zfill(3) + ".pt"
        if (t+1) % 5 == 0:
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved PyTorch Model checkpoint to {checkpoint_filename}")
    print("Done!")
