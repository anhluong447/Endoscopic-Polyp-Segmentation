import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


def weights_init(model):
    """Initializes linear layer weights using Xavier uniform distribution."""
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)


def save_model(model, optimizer, path):
    """Saves checkpoint containing state dict of model and optimizer."""
    # Handle DataParallel state dict saving
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_model(model, optimizer, path, device="cpu"):
    """Loads checkpoint, handling potential DataParallel prefixes (module.)."""
    checkpoint = torch.load(path, map_location=device)
    
    state_dict = checkpoint["model"]
    new_state_dict = OrderedDict()
    
    # Strip 'module.' prefix if it exists in checkpoint but not in current model
    is_model_dp = isinstance(model, nn.DataParallel)
    
    for k, v in state_dict.items():
        name = k
        if k.startswith("module.") and not is_model_dp:
            name = k[7:]  # Remove 'module.'
        elif not k.startswith("module.") and is_model_dp:
            name = f"module.{k}"  # Add 'module.'
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            
    return model, optimizer


def rle_to_string(runs):
    """Converts a sequence of run-length numbers to a string."""
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    """Encodes a single binary mask into a RLE string representation."""
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)
