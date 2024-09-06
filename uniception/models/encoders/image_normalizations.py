# --------------------------------------------------------
# Image normalizations for the different encoders
# Encoders defined in UniCeption must have their corresponding image normalization defined here.
# --------------------------------------------------------
import torch
from dataclasses import dataclass


@dataclass
class ImageNormalization:
    mean: torch.Tensor
    std: torch.Tensor


IMAGE_NORMALIZATION_DICT = {
    "dummy": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "croco": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dust3r": ImageNormalization(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
    "dinov2": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "radio": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
}
