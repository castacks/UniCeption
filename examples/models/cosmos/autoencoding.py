import os

import cv2
import torch
from matplotlib import pyplot as plt

from uniception.models.encoders.base import ViTEncoderInput
from uniception.models.encoders.cosmos import CosmosEncoder
from uniception.models.prediction_heads.cosmos import CosmosSingleChannel

base_path = os.path.dirname(os.path.abspath(__file__))

encoder = CosmosEncoder(
    name="cosmos",
    patch_size=8,
    pretrained_checkpoint_path=os.path.join(
        base_path, "../../../checkpoints/encoders/cosmos/Cosmos-Tokenizer-CI8x8/encoder.pth"
    ),
)

decoder = CosmosSingleChannel(
    patch_size=8,
    pretrained_checkpoint_path=os.path.join(base_path, "../../../checkpoints/prediction_heads/cosmos/decoder_8.pth"),
)

example_image = cv2.imread(os.path.join(base_path, "./example.png"))
example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
example_tensor = torch.tensor(example_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
example_tensor = example_tensor * 2.0 - 1.0  # Normalize to [-1, 1] according to the COSMOS Encoder

encoded_latent = encoder(ViTEncoderInput("cosmos", example_tensor)).features

decoded_image = decoder(encoded_latent)
decoded_image = (decoded_image + 1.0) / 2.0  # Denormalize to [0, 1] for visualization

# plot the original and decoded images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(example_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(decoded_image.squeeze().detach().permute(1, 2, 0).cpu().numpy())
plt.title("Decoded Image")
plt.axis("off")

plt.savefig(os.path.join(base_path, "example_decoded.png"))
