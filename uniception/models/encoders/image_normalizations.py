# --------------------------------------------------------
# Image normalizations for the different encoders
# Encoders defined in UniCeption must have their corresponding image normalization defined here.
# --------------------------------------------------------

import torch

IMAGE_NORMALIZATION_DICT = {"dummy": (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))}
