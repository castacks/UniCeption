import os

os.environ["TORCH_LOGS"] = "recompiles"
# os.environ["PYTORCH_JIT_LOG_SCHEMA"] = "1"

import torch
from uniception.models.encoders import encoder_factory, ViTEncoderInput
from itertools import product

if __name__ == "__main__":

    # Test the dinov2 encoder
    model = encoder_factory("dinov2", size="large", name="encoder", disable_torch_compile_for_pe=True).cuda()
    model.compile(
        dynamic=True,
        fullgraph=False,
        options={
            "shape_padding": True,
        },
    )

    print(model)

    # Create a dummy input tensor
    hw_list = [(int(x * 14), int(y * 14)) for (x, y) in product(range(16, 24), range(16, 24))]

    dummy_input = [ViTEncoderInput(image=torch.randn(16, 3, *hw).cuda(), data_norm_type="dinov2") for hw in hw_list]

    for i in range(100):
        # Forward pass through the model
        for x in dummy_input:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(x)
                print(x.image.shape)
