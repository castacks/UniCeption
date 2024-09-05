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
    "dummy": ImageNormalization(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])),
    "croco": ImageNormalization(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
    "dust3r": ImageNormalization(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])),
    "dinov2": ImageNormalization(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
}
