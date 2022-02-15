"""
Script associated with the Attention Complexity Sweep. This experiment measures inference time with an increasing
number of tokens in multiple attention layers. New attention layers should be added to the attention_layer dict.

To run:
    1.- Start sweep using
        wandb sweep sweep_attention_complexity.yml
    2.- Instantiate an agent of the started sweep in a directory where this file is located
        wandb agent <username>/<project-name>/<sweep-id>
"""
import timm
import wandb
import torch
import warnings
import performer_pytorch
import numpy as np

# Hyperparameters and config
# Input
num_tokens = 576
dim = 768
# Model architecture
attention_variant = None  # "performer"
attention_heads = 12
mixed_precision = False
# Config
config_dict = {
    "mixed_precision": mixed_precision,

    "attention_variant": attention_variant,
    "attention_heads": attention_heads,
    "num_tokens": num_tokens,
    "dim": dim,
}

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    # Init wandb
    wandb.init(config=config_dict)
    config = wandb.config
    # Re-read config for wandb-sweep-managed inference
    mixed_precision = config["mixed_precision"]
    attention_variant = config["attention_variant"]
    attention_heads = config["attention_heads"]
    num_tokens = config["num_tokens"]
    dim = config["dim"]

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # Create attention layer
    attention_layer = {
        "None": timm.models.vision_transformer.Attention(dim, attention_heads, True),
        "performer": performer_pytorch.SelfAttention(dim=dim, heads=attention_heads, causal=False),
    }
    attention = attention_layer[attention_variant].to(device)

    n_inferences = 500
    wandb.log({"num_inferences": n_inferences})
    measures = np.zeros((n_inferences, 1))
    x = torch.rand(1, num_tokens, dim).to(device)

    # Cuda events
    t0 = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Measure inference time
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            _ = attention(x)  # Warm-up
            for i in range(n_inferences):
                t0.record()
                y = attention(x)
                end.record()
                torch.cuda.synchronize()
                measures[i] = t0.elapsed_time(end)
    mean_ms = np.mean(measures)
    std_ms = np.std(measures)

    # Log and print results
    wandb.log({"ms": mean_ms, "std_ms": std_ms})
    print(f"Inference speed (ms): {mean_ms:.4f} +- {std_ms:.4f}")
