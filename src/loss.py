import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.losses import one_hot


class CEDiceLoss(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super(CEDiceLoss, self).__init__()
        self.eps: float = 1e-6
        # Ensure weights is a 1D tensor
        self.weights: torch.Tensor = weights.view(-1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
        
        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")
        
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError(f"input and target spatial shapes must be the same. Got: {input.shape} and {target.shape}")
        
        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device}, {target.device}")
        
        if not self.weights.shape[0] == input.shape[1]:
            raise ValueError(f"The number of weights ({self.weights.shape[0]}) must equal the number of classes ({input.shape[1]})")
        
        if not abs(torch.sum(self.weights).item() - 1.0) < 1e-3:
            raise ValueError(f"The sum of all weights must equal 1 (got {torch.sum(self.weights).item()})")
            
        # Cross entropy loss (needs 1D weights)
        celoss = nn.CrossEntropyLoss(self.weights)(input, target)
        
        # Softmax over classes dimension
        input_soft = F.softmax(input, dim=1)

        # Create target one-hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # Compute dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        
        # Apply weights along classes dimension
        weights_expanded = self.weights.view(1, -1)  # shape: (1, C)
        dice_score = torch.sum(dice_score * weights_expanded, dim=1)
        
        return torch.mean(1. - dice_score) + celoss
