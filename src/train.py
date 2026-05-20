import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

from src.dataset import SegDataClass
from src.models import UNet
from src.loss import CEDiceLoss
from src.utils import weights_init, save_model, load_model


def train_epoch(model, dataloader, loss_function, optimizer, device, epoch, display_step):
    model.train()
    train_loss = 0.0
    
    for i, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss = loss_function(outputs, targets.long())
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (i + 1) % display_step == 0:
            print(f"Train Epoch: {epoch + 1} [{ (i + 1) * len(data)}/{len(dataloader.dataset)} "
                  f"({100. * (i + 1) * len(data) / len(dataloader.dataset):.1f}%)]\tLoss: {loss.item():.4f}")
            
    return train_loss / (i + 1)


def validate(model, dataloader, loss_function, device):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = loss_function(outputs, targets.long())
            val_loss += loss.item()
            
    return val_loss / (i + 1)


def train_model(
    images_path,
    masks_path,
    epochs=30,
    batch_size=12,
    learning_rate=1e-4,
    checkpoint_path="unet_model.pth",
    pretrained_path=None,
    use_wandb=False,
    wandb_project="PolypSegment",
    wandb_key=None,
    display_step=50,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Set up datasets
    dataset = SegDataClass(images_path, masks_path)
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    
    torch.manual_seed(42)
    train_set, valid_set = random_split(dataset, [train_size, valid_size])
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = UNet(n_class=3)
    
    # Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    
    # Try to load pretrained model
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained checkpoint from {pretrained_path}...")
        try:
            model, optimizer = load_model(model, optimizer, pretrained_path, device=device)
        except Exception as e:
            print(f"Error loading checkpoint, initializing weights: {e}")
            model.apply(weights_init)
    else:
        print("Initializing weights from scratch...")
        model.apply(weights_init)
        
    # DataParallel wrapper if CUDA is available and multiple GPUs are detected
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # Loss function with weights (background, polyp, boundary)
    # The weights must sum to 1.0 (0.4 + 0.55 + 0.05 = 1.0)
    weights = torch.Tensor([[0.4, 0.55, 0.05]]).to(device)
    loss_function = CEDiceLoss(weights)
    
    # Scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)
    
    # Initialize WandB
    if use_wandb:
        import wandb
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(project=wandb_project, config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "pretrained": pretrained_path is not None
        })
        
    print("Starting training...")
    best_loss = float("inf")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nStart epoch #{epoch + 1}, learning rate for this epoch: {current_lr:.6f}")
        
        # Train and validate
        train_loss = train_epoch(model, train_dataloader, loss_function, optimizer, device, epoch, display_step)
        val_loss = validate(model, valid_dataloader, loss_function, device)
        
        # Step the scheduler
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Done epoch #{epoch + 1}, time: {elapsed:.1f}s, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "lr": current_lr
            })
            
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving checkpoint to {checkpoint_path}")
            save_model(model, optimizer, checkpoint_path)
            
    print("Training finished.")
    if use_wandb:
        wandb.finish()
