import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToPILImage, PILToTensor, Compose, InterpolationMode

from src.dataset import UNetTestDataClass
from src.models import UNet
from src.utils import load_model, rle_encode_one_mask


def run_inference(
    test_images_path,
    model_path,
    output_dir="predicted_masks",
    submission_path="output.csv",
    batch_size=8,
    visualize_count=5,
    visualize_path="visualization.png",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Set up directory for saving predictions
    os.makedirs(output_dir, exist_ok=True)

    # Dataloader
    dataset = UNetTestDataClass(test_images_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = UNet(n_class=3)
    
    # Load model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
    print(f"Loading checkpoint from {model_path}...")
    model, _ = load_model(model, None, model_path, device=device)
    
    # If multiple GPUs, wrap model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    model.eval()

    # Visualization
    if visualize_count > 0:
        print(f"Generating visualization plot to {visualize_path}...")
        # Get one batch for visualization
        first_batch = next(iter(dataloader))
        img, img_paths, h_orig, w_orig = first_batch
        
        # Limit count to batch size or requested count
        actual_count = min(visualize_count, len(img))
        
        with torch.no_grad():
            predict = model(img.to(device))
            
        fig, arr = plt.subplots(actual_count, 2, figsize=(12, 4 * actual_count))
        if actual_count == 1:
            arr = np.expand_dims(arr, axis=0)
            
        arr[0][0].set_title('Image')
        arr[0][1].set_title('Predict')
        
        for i in range(actual_count):
            # Plot original image (C, H, W) -> (H, W, C)
            orig_img_np = img[i].permute(1, 2, 0).cpu().numpy()
            arr[i][0].imshow(orig_img_np)
            arr[i][0].axis('off')
            
            # Predict one hot mapping
            pred_class = torch.argmax(predict[i], 0)
            pred_one_hot = F.one_hot(pred_class, num_classes=3).float().cpu().numpy()
            arr[i][1].imshow(pred_one_hot)
            arr[i][1].axis('off')
            
        plt.tight_layout()
        plt.savefig(visualize_path)
        plt.close()
        print(f"Visualization saved.")

    # Run prediction and save masks
    print("Predicting masks on test images...")
    
    resize_nearest = lambda h, w: Resize((h, w), interpolation=InterpolationMode.NEAREST)
    to_pil = ToPILImage()
    
    for batch_idx, (imgs, paths, h_orig, w_orig) in enumerate(dataloader):
        imgs = imgs.to(device)
        with torch.no_grad():
            predicted_masks = model(imgs)
            
        for i in range(len(paths)):
            img_id = os.path.basename(paths[i]).split('.')[0]
            filename = f"{img_id}.png"
            out_mask_path = os.path.join(output_dir, filename)
            
            # Convert class predictions to one hot (512, 512, 3) -> (3, 512, 512)
            pred_class = torch.argmax(predicted_masks[i], 0)
            pred_one_hot = F.one_hot(pred_class, num_classes=3).permute(2, 0, 1).float()
            
            # Convert to PIL image
            pil_mask = to_pil(pred_one_hot)
            
            # Resize back to original dimensions using nearest interpolation
            h_val, w_val = h_orig[i].item(), w_orig[i].item()
            resized_mask = resize_nearest(h_val, w_val)(pil_mask)
            
            # Save mask
            resized_mask.save(out_mask_path)
            
    print(f"All masks saved to {output_dir}")

    # Convert saved masks to RLE format for submission
    print("Generating submission file...")
    ids = []
    strings = []
    
    for filename in sorted(os.listdir(output_dir)):
        if not filename.endswith('.png'):
            continue
            
        img_id = filename.split('.')[0]
        filepath = os.path.join(output_dir, filename)
        
        # Read saved image in BGR and convert to RGB
        img = cv2.imread(filepath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Encode channel 0 (polyp) and channel 1 (boundary)
        for channel in range(2):
            ids.append(f"{img_id}_{channel}")
            rle_str = rle_encode_one_mask(img[:, :, channel])
            strings.append(rle_str)
            
    df = pd.DataFrame({
        'Id': ids,
        'Expected': strings
    })
    df.to_csv(submission_path, index=False)
    print(f"Submission file saved successfully to {submission_path}")
