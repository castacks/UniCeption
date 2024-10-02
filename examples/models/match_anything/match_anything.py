"""
Example of using the match-anything model.
"""

import os

import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from uniception.models.factory import MatchAnythingModel


def warp_image_with_flow(source_image, source_mask, target_image, flow):
    """
    Warp the target to source image using the given flow vectors.
    Flow vectors indicate the displacement from source to target.
    """
    # assert source_image.shape[-1] == 3
    # assert target_image.shape[-1] == 3

    assert flow.shape[-1] == 2

    # Get the shape of the source image
    height, width = source_image.shape[:2]

    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply flow displacements
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    x_new = np.clip(x + flow_x, 0, width - 1) + 0.5
    y_new = np.clip(y + flow_y, 0, height - 1) + 0.5

    x_new = (x_new / target_image.shape[1]) * 2 - 1
    y_new = (y_new / target_image.shape[0]) * 2 - 1

    warped_image = F.grid_sample(
        torch.from_numpy(target_image).permute(2, 0, 1)[None, ...].float(),
        torch.from_numpy(np.stack([x_new, y_new], axis=-1)).float()[None, ...],
        mode="bilinear",
        align_corners=False,
    )

    warped_image = warped_image[0].permute(1, 2, 0).numpy()

    if source_mask is not None:
        warped_image = warped_image * (source_mask > 0.5)

    return warped_image


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    #                                                                             #
    #                NOTICE: Confidence is not trained in this ckpt               #
    #                                                                             #
    ###############################################################################

    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(os.path.dirname(current_file_path), "../../../checkpoints")
    ma_224_dpt_ckpt_path = os.path.join(
        relative_checkpoint_path, "examples", "match_anything", "ma_224_dpt_uniception.ckpt"
    )

    # equivalent to MatchAnythingModel(**ckpt["model_args"]).load_state_dict(ckpt["model"], strict=strict)
    model = MatchAnythingModel.from_pretrained(ma_224_dpt_ckpt_path, strict=True)
    model.to(device)

    img0 = Image.open(os.path.join(__file__, "..", "img0.png"))
    img1 = Image.open(os.path.join(__file__, "..", "img1.png"))

    img0_tensor = (torch.from_numpy(np.array(img0))[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255).to(device)
    img1_tensor = (torch.from_numpy(np.array(img1))[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255).to(device)

    # Normalize images according to Encoder's requirement
    data_norm_type = model.encoder.data_norm_type
    image_normalization = IMAGE_NORMALIZATION_DICT[data_norm_type]
    img_mean = image_normalization.mean.to(device).view(1, 3, 1, 1)
    img_std = image_normalization.std.to(device).view(1, 3, 1, 1)
    print(
        "Using data normalization type:",
        data_norm_type,
        "with mean:",
        img_mean.flatten(),
        "and std:",
        img_std.flatten(),
    )

    img0_tensor = (img0_tensor - img_mean) / img_std
    img1_tensor = (img1_tensor - img_mean) / img_std

    # Run a forward pass
    view1 = {"img": img0_tensor, "instance": [0], "data_norm_type": data_norm_type}
    view2 = {"img": img1_tensor, "instance": [1], "data_norm_type": data_norm_type}

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            result = model(view1, view2)

    flow = result["flow"]["flow_output"]
    flow_conf = result["flow"]["flow_output_conf"]  # Notice! confidence is not trained in this model
    non_occluded_mask = result["occlusion"]["mask"]
    non_occluded_logits = result["occlusion"]["logits"]

    # first half the batch = forward flow/occlusion , second half = backward flow/occlusion
    flow_fwd, flow_bwd = flow.chunk(2, dim=0)
    flow_conf_fwd, flow_conf_bwd = flow_conf.chunk(2, dim=0)
    non_occluded_mask_fwd, non_occluded_mask_bwd = non_occluded_mask.chunk(2, dim=0)
    non_occluded_logits_fwd, non_occluded_logits_bwd = non_occluded_logits.chunk(2, dim=0)

    # postprocess - mask all region where network think is occluded
    flow_fwd[(non_occluded_mask_fwd < 0.5).repeat(1, 2, 1, 1)] = 0
    flow_bwd[(non_occluded_mask_bwd < 0.5).repeat(1, 2, 1, 1)] = 0

    # Visualize the results
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0, 0].imshow(img0)
    axs[0, 0].set_title("Image 1")

    axs[1, 0].imshow(img1)
    axs[1, 0].set_title("Image 2")

    flow_fwd = flow_fwd[0].cpu().numpy().transpose(1, 2, 0)
    axs[0, 2].imshow(flow_vis.flow_to_color(flow_fwd))
    axs[0, 2].set_title("Forward Flow")

    flow_bwd = flow_bwd[0].cpu().numpy().transpose(1, 2, 0)
    axs[1, 2].imshow(flow_vis.flow_to_color(flow_bwd))
    axs[1, 2].set_title("Backward Flow")

    warp_2to1 = warp_image_with_flow(
        source_image=img0_tensor[0].cpu().numpy().transpose(1, 2, 0),
        source_mask=non_occluded_mask_fwd[0].cpu().numpy().transpose(1, 2, 0),
        target_image=img1_tensor[0].cpu().numpy().transpose(1, 2, 0),
        flow=flow_fwd,
    )
    axs[0, 1].imshow(warp_2to1)
    axs[0, 1].set_title("Warp 2 to 1, should look like Image 1")

    warp_1to2 = warp_image_with_flow(
        source_image=img1_tensor[0].cpu().numpy().transpose(1, 2, 0),
        source_mask=non_occluded_mask_bwd[0].cpu().numpy().transpose(1, 2, 0),
        target_image=img0_tensor[0].cpu().numpy().transpose(1, 2, 0),
        flow=flow_bwd,
    )
    axs[1, 1].imshow(warp_1to2)
    axs[1, 1].set_title("Warp 1 to 2, should look like Image 2")

    fig_path = os.path.join(os.path.dirname(__file__), "result.png")
    fig.savefig(fig_path)
    print(f"Results saved to {fig_path}")
