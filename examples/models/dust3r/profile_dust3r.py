import torch
from dust3r import get_model_configurations_and_checkpoints
from uniception.models.factory import DUSt3R
from uniception.utils.profile import benchmark_torch_function

if __name__ == "__main__":
    # Get model configurations and checkpoints
    model_configurations, model_to_checkpoint_path = get_model_configurations_and_checkpoints()

    # Test different DUSt3R models using UniCeption modules
    for model_name in model_configurations:
        dust3r_model = DUSt3R(
            name=model_name,
            img_size=(512, 512) if "512" in model_name else (224, 224),
            patch_embed_cls="PatchEmbedDust3R",
            pred_head_type="linear" if "linear" in model_name else "dpt",
            pretrained_checkpoint_path=model_to_checkpoint_path[model_name]["ckpt_path"],
        )
        print(f"DUSt3R model ({model_name}) initialized successfully!")

        # Initialize device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dust3r_model.to(device)
        print(f"Running on {device}")

        # Generate random input tensors
        img_size = (512, 512) if "512" in model_name else (224, 224)
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            # Prepare input views
            view1_instances = range(batch_size)
            view1_img_tensor = torch.randn(batch_size, 3, *img_size).to(device)
            view1 = {"img": view1_img_tensor, "instance": view1_instances, "data_norm_type": "dust3r"}
            view2_instances = range(batch_size)
            view2_instances = [id + batch_size for id in view2_instances]
            view2_img_tensor = torch.randn(batch_size, 3, *img_size).to(device)
            view2 = {"img": view2_img_tensor, "instance": view2_instances, "data_norm_type": "dust3r"}

            # Benchmark the forward pass of the model
            with torch.no_grad():
                with torch.autocast("cuda", enabled=True):
                    execution_time = benchmark_torch_function(dust3r_model, view1, view2)
                    print(
                        f"\033[92mForward pass for {model_name}, batch size : {batch_size} completed in {execution_time:.3f} milliseconds\033[0m"
                    )
