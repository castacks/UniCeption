import os

import torch
from match_anything import MatchAnythingModel

from uniception.utils.profile import benchmark_torch_function

if __name__ == "__main__":
    # Initialize device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(os.path.dirname(current_file_path), "../../../checkpoints")
    ma_224_dpt_ckpt_path = os.path.join(
        relative_checkpoint_path, "examples", "match_anything", "ma_224_dpt_uniception.ckpt"
    )

    model = MatchAnythingModel.from_pretrained(ma_224_dpt_ckpt_path, use_single_head=True, strict=False)
    model.to(device)
    print(f"Running on {device}")

    # Generate random input tensors
    img_size = (224, 224)
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        # Prepare input views
        view1_instances = range(batch_size)
        view1_img_tensor = torch.randn(batch_size, 3, *img_size).to(device)
        data_norm_type = model.encoder.data_norm_type
        view1 = {"img": view1_img_tensor, "instance": view1_instances, "data_norm_type": data_norm_type}
        view2_instances = range(batch_size, 2 * batch_size)
        view2_instances = [id + batch_size for id in view2_instances]
        view2_img_tensor = torch.randn(batch_size, 3, *img_size).to(device)
        view2 = {"img": view2_img_tensor, "instance": view2_instances, "data_norm_type": data_norm_type}

        with torch.no_grad():
            # Benchmark the forward pass of the model
            with torch.autocast("cuda", torch.bfloat16, enabled=True):
                execution_time = benchmark_torch_function(model, view1, view2)
                print(
                    f"\033[92mForward pass for batch size : {batch_size} completed in {execution_time:.3f} milliseconds\033[0m"
                )
