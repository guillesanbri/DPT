import os
import wandb
import torch
import warnings
import numpy as np
import torchvision.transforms
from fvcore.nn import FlopCountAnalysis

from dpt.models import DPTDepthModel


def get_flops(model, x, unit="G", quiet=True):
    _prefix = {'k': 1e3,  # kilo
               'M': 1e6,  # mega
               'G': 1e9,  # giga
               'T': 1e12,  # tera
               'P': 1e15,  # peta
               }
    flops = FlopCountAnalysis(model, x)
    num_flops = flops.total() / _prefix[unit]
    if not quiet:
        print(f"Model FLOPs: {num_flops:.2f} {unit}FLOPs")
    return num_flops


def get_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    model_size = os.path.getsize("tmp.pt")/1e6
    os.remove('tmp.pt')
    return model_size


# Hyperparameters and config
# Input
net_w, net_h = 640, 192
h_kitti, w_kitti = 352, 1216
# Model architecture
backbone = "vitb_rn50_384"  # "vitb_effb0"
transformer_hooks = "str:8,11"
attention_variant = None  # "performer"
attention_heads = 12
mixed_precision = False

config_dict = {
    "input_size": f"{net_h},{net_w}",
    "downsampling": "Resize image along w and h",
    "mixed_precision": mixed_precision,

    "backbone": backbone,
    "transformer_hooks": transformer_hooks,
    "attention_variant": attention_variant,
    "attention_heads": attention_heads,
}

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    # Init wandb
    wandb.init(config=config_dict)
    config = wandb.config
    # Re-read config for wandb-sweep-managed inference
    mixed_precision = config["mixed_precision"]
    backbone = config["backbone"]
    transformer_hooks = config["transformer_hooks"]
    attention_variant = config["attention_variant"]
    if attention_variant == "None":
        attention_variant = None
    attention_heads = config["attention_heads"]
    input_size = config["input_size"]
    net_h = int(input_size.split(",")[0])
    net_w = int(input_size.split(",")[1])

    # Convert str hooks to list (wandb hacky solution to display hooks correctly)
    assert isinstance(transformer_hooks, str) and transformer_hooks[:4] == "str:", \
        'Hooks are not in the format "str:[att_hook1, att_hook2]"'
    conv_hooks = {"vitb_rn50_384": [0, 1], "vitb_effb0": [1, 2]}[backbone]
    transformer_hooks = [int(hook) for hook in transformer_hooks.split(":")[-1].split(",")]
    hooks = conv_hooks + transformer_hooks

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Create model
    model = DPTDepthModel(
                path=None,
                scale=0.00006016,  # KITTI
                shift=0.00579,
                invert=True,
                backbone=backbone,
                attention_heads=attention_heads,
                hooks=hooks,
                non_negative=True,
                enable_attention_hooks=False,
                attention_variant=attention_variant).to(device)

    n_inferences = 500
    wandb.log({"num_inferences": n_inferences})
    measures = np.zeros((n_inferences, 1))
    x = torch.rand(1, 3, h_kitti, w_kitti).to(device)
    print(f"Kitti size: {h_kitti}, {w_kitti} | Network input size: {net_h}, {net_w}")

    # Cuda events
    t0 = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Measure inference time
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            dummy = torchvision.transforms.Resize((net_h, net_w))(x)
            _ = model(dummy)  # Warm-up
            for i in range(n_inferences):
                t0.record()
                if net_h != h_kitti or net_w != w_kitti:
                    x = torchvision.transforms.Resize((net_h, net_w))(x)
                y = model(x)
                if net_h != h_kitti or net_w != w_kitti:
                    _ = torch.nn.functional.interpolate(y.unsqueeze(1),
                                                        size=(h_kitti, w_kitti),
                                                        mode="bicubic",
                                                        align_corners=True)
                end.record()
                torch.cuda.synchronize()
                measures[i] = t0.elapsed_time(end)
    mean_ms = np.mean(measures)
    std_ms = np.std(measures)
    fps = 1000/measures
    mean_fps = np.mean(fps)
    std_fps = np.std(fps)

    GFLOPs = get_flops(model.to("cpu"), x.to("cpu"))
    model_MB = get_model_size(model)
    wandb.log({"FPS": mean_fps, "std_fps": std_fps, "ms": mean_ms, "std_ms": std_ms, "GFLOPs": GFLOPs, "MB": model_MB})

    print(f"FPS: {mean_fps:.2f} +- {1/std_fps:.2f} || Inference speed (ms): {mean_ms:.4f} +- {std_ms:.4f}")
    print(f"GFLOPs: {GFLOPs:.3f} || Model size (MB): {model_MB:.2f}")
