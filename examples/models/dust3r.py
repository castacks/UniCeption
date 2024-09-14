"""
Initalizing Pre-trained DUSt3R/MASt3R using UniCeption
"""

import os
import torch
import torch.nn as nn
from typing import List, Tuple

from uniception.models.encoders import ViTEncoderInput
from uniception.models.encoders.croco import CroCoEncoder
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from uniception.models.libs.croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D
from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformerIFR,
    MultiViewCrossAttentionTransformerInput,
)
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.base import AdaptorInput, PredictionHeadLayeredInput
from uniception.models.prediction_heads.adaptors import PointMapAdaptor, ConfidenceAdaptor


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}


def transpose_to_landscape(head, activate=True):
    """Predict in the correct aspect-ratio,
    then transpose the result in landscape
    and stack everything back together.
    """

    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), "true_shape must be all identical"
        H, W = true_shape[0].cpu().tolist()
        # res = head(decout, (H, W))
        res = head(decout)
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = width >= height
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            # return head(decout, (H, W))
            return head(decout)
        if is_portrait.all():
            # return transposed(head(decout, (W, H)))
            return transposed(head(decout))

        # batch is a mix of both portraint & landscape
        def selout(ar):
            return [d[ar] for d in decout]

        # l_result = head(selout(is_landscape), (H, W))
        # p_result = transposed(head(selout(is_portrait), (W, H)))
        l_result = head(selout(is_landscape))
        p_result = transposed(head(selout(is_portrait)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def is_symmetrized(gt1, gt2):
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


class DUSt3R(nn.Module):
    "DUSt3R defined with UniCeption Modules"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "dust3r",
        img_size: tuple = (224, 224),
        patch_embed_cls: str = "PatchEmbedDust3R",
        pred_head_type: str = "linear",
        pred_head_output_dim: int = 4,
        pred_head_feature_dim: int = 256,
        depth_mode: Tuple[str, float, float] = ("exp", -float("inf"), float("inf")),
        conf_mode: Tuple[str, float, float] = ("exp", 1, float("inf")),
        pos_embed: str = "RoPE100",
        landscape_only=True,
        pretrained_encoder_checkpoint_path: str = None,
        pretrained_decoder_checkpoint_path: str = None,
        pretrained_pred_head_checkpoint_paths: List[str] = None,
        pretrained_pred_head_regressor_checkpoint_paths: List[str] = None,
        *args,
        **kwargs,
    ):
        """
        Two-view model containing siamese encoders followed by a two-view cross-attention transformer and respective downstream heads.
        The goal is to output scene representation directly, both images in view1's frame (hence the asymmetry).

        Args:
            name (str): Name of the model.
            data_norm_type (str): Type of data normalization. (default: "dust3r")
            img_size (tuple): Size of input images. (default: (224, 224))
            patch_embed_cls (str): Class for patch embedding. (default: "PatchEmbedDust3R"). Options:
            - "PatchEmbedDust3R"
            - "ManyAR_PatchEmbed"
            pred_head_type (str): Type of prediction head. (default: "linear"). Options:
            - "linear"
            - "dpt"
            pred_head_output_dim (int): Output dimension of prediction head. (default: 4)
            pred_head_feature_dim (int): Feature dimension of prediction head. (default: 256)
            depth_mode (Tuple[str, float, float]): Depth mode settings (mode=['linear', 'square', 'exp'], vmin, vmax). (default: ('exp', -inf, inf))
            conf_mode (Tuple[str, float, float]): Confidence mode settings (mode=['linear', 'square', 'exp'], vmin, vmax). (default: ('exp', 1, inf))
            pos_embed (str): Position embedding type. (default: 'RoPE100')
            landscape_only (bool): Run downstream head only in landscape orientation. (default: True)
            pretrained_encoder_checkpoint_path (str): Path to pretrained encoder checkpoint. (default: None)
            pretrained_decoder_checkpoint_path (str): Path to pretrained decoder checkpoint. (default: None)
            pretrained_pred_head_checkpoint_paths (List[str]): Paths to pretrained prediction head checkpoints. (default: None)
            pretrained_pred_head_regressor_checkpoint_paths (List[str]): Paths to pretrained prediction head regressor checkpoints. (default: None)
        """
        super().__init__(*args, **kwargs)

        # Initialize RoPE for the CroCo Encoder & Multi-View Cross Attention Transformer
        self.pos_embed = pos_embed
        freq = float(pos_embed[len("RoPE") :])
        self.rope = RoPE2D(freq=freq)

        # Initialize Encoder
        self.encoder = CroCoEncoder(
            name=name,
            data_norm_type=data_norm_type,
            patch_embed_cls=patch_embed_cls,
            img_size=img_size,
            pretrained_checkpoint_path=pretrained_encoder_checkpoint_path,
        )

        # Initialize Multi-View Cross Attention Transformer
        self.decoder = MultiViewCrossAttentionTransformerIFR(
            name="base_decoder",
            input_embed_dim=self.encoder.enc_embed_dim,
            num_views=2,
            indices=[6, 9],
            norm_intermediate=False,
            custom_positional_encoding=self.rope,
            pretrained_checkpoint_path=pretrained_decoder_checkpoint_path,
        )

        # Initialize Prediction Heads
        self.landscape_only = landscape_only
        if pred_head_type == "linear":
            pass
        elif pred_head_type == "dpt":
            # Initialze Predction Head 1
            self.dpt_feature_head1 = DPTFeature(
                patch_size=self.encoder.patch_size,
                hooks=[0, 1, 2, 3],
                input_feature_dims=[self.encoder.enc_embed_dim] + [self.decoder.dim] * 3,
                feature_dim=pred_head_feature_dim,
            )
            print(
                self.dpt_feature_head1.load_state_dict(
                    torch.load(pretrained_pred_head_checkpoint_paths[0], weights_only=False)
                )
            )
            self.dpt_regressor_head1 = DPTRegressionProcessor(
                input_feature_dim=pred_head_feature_dim,
                output_dim=pred_head_output_dim,
            )
            print(
                self.dpt_regressor_head1.load_state_dict(
                    torch.load(pretrained_pred_head_regressor_checkpoint_paths[0], weights_only=False)
                )
            )
            self.downstream_head1 = nn.Sequential(self.dpt_feature_head1, self.dpt_regressor_head1)
            # Initialize Prediction Head 2
            self.dpt_feature_head2 = DPTFeature(
                patch_size=self.encoder.patch_size,
                hooks=[0, 1, 2, 3],
                input_feature_dims=[self.encoder.enc_embed_dim] + [self.decoder.dim] * 3,
                feature_dim=pred_head_feature_dim,
            )
            print(
                self.dpt_feature_head2.load_state_dict(
                    torch.load(pretrained_pred_head_checkpoint_paths[1], weights_only=False)
                )
            )
            self.dpt_regressor_head2 = DPTRegressionProcessor(
                input_feature_dim=pred_head_feature_dim,
                output_dim=pred_head_output_dim,
            )
            print(
                self.dpt_regressor_head2.load_state_dict(
                    torch.load(pretrained_pred_head_regressor_checkpoint_paths[1], weights_only=False)
                )
            )
            self.downstream_head2 = nn.Sequential(self.dpt_feature_head2, self.dpt_regressor_head2)
            # Magic wrapper to handle landscape/portrait
            self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
            self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

        # Initialize Adaptors
        self.point_map_adaptor = PointMapAdaptor(
            name="depth",
            mode=depth_mode[0],
            vmin=depth_mode[1],
            vmax=depth_mode[2],
        )
        self.confidence_adaptor = ConfidenceAdaptor(
            name="confidence",
            confidence_type=conf_mode[0],
            vmin=conf_mode[1],
            vmax=conf_mode[2],
        )

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, data_norm_type):
        if img1.shape[-2:] == img2.shape[-2:]:
            encoder_input = ViTEncoderInput(image=torch.cat((img1, img2), dim=0), data_norm_type=data_norm_type)
            encoder_input.true_shape = torch.cat((true_shape1, true_shape2), dim=0)
            encoder_output = self.encoder(encoder_input)
            out, out2 = encoder_output.features.chunk(2, dim=0)
        else:
            encoder_input = ViTEncoderInput(image=img1, data_norm_type=data_norm_type)
            encoder_input.true_shape = true_shape1
            out = self.encoder(encoder_input)
            out = out.features
            encoder_input2 = ViTEncoderInput(image=img2)
            encoder_input2.true_shape = true_shape2
            out2 = self.encoder(encoder_input2)
            out2 = out2.features
        return out, out2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get("true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get("true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # Warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # Computing half of forward pass!'
            feat1, feat2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2], data_norm_type=view1["data_norm_type"]
            )
            feat1, feat2 = interleave(feat1, feat2)
        else:
            feat1, feat2 = self._encode_image_pairs(img1, img2, shape1, shape2, data_norm_type=view1["data_norm_type"])

        return (shape1, shape2), (feat1, feat2)

    def _downstream_head(self, head_num, decout, img_shape):
        head = getattr(self, f"head{head_num}")
        head_input = PredictionHeadLayeredInput(
            list_features=decout[f"{head_num}"], target_output_shape=img_shape[0].cpu().int().tolist()
        )
        return head(head_input, img_shape)

    def forward(self, view1, view2):
        # Encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2) = self._encode_symmetrized(view1, view2)

        # Combine all ref images into view-centric representation
        decoder_input = MultiViewCrossAttentionTransformerInput(features=[feat1, feat2])
        final_decoder_multi_view_feat, intermediate_decoder_multi_view_feat = self.decoder(decoder_input)

        # Define DPT feature list
        decoder_outputs = {
            "1": [
                feat1,
                intermediate_decoder_multi_view_feat[0].features[0],
                intermediate_decoder_multi_view_feat[1].features[0],
                final_decoder_multi_view_feat.features[0],
            ],
            "2": [
                feat2,
                intermediate_decoder_multi_view_feat[0].features[1],
                intermediate_decoder_multi_view_feat[1].features[1],
                final_decoder_multi_view_feat.features[1],
            ],
        }

        # Downstream prediction head
        with torch.autocast("cuda", enabled=False):
            res1 = self._downstream_head(1, decoder_outputs, shape1)
            res2 = self._downstream_head(2, decoder_outputs, shape2)

        # Get all the features and slice for processing
        res1_all_features = res1.decoded_channels
        res2_all_features = res2.decoded_channels
        res1_pointmap_features, res1_confidence_features = torch.split(res1_all_features, [3, 1], dim=1)
        res2_pointmap_features, res2_confidence_features = torch.split(res2_all_features, [3, 1], dim=1)

        # Run adaptors
        output1 = {}
        output2 = {}
        output1["pts3d"] = self.point_map_adaptor(AdaptorInput(res1_pointmap_features, shape1[0]))
        output1["conf"] = self.confidence_adaptor(AdaptorInput(res1_confidence_features, shape1[0]))
        output2["pts3d"] = self.point_map_adaptor(AdaptorInput(res2_pointmap_features, shape2[0]))
        output2["conf"] = self.confidence_adaptor(AdaptorInput(res2_confidence_features, shape2[0]))
        return output1, output2


def log_data_to_rerun(image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name):
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
            from_parent=False,
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]
    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(os.path.dirname(current_file_path), "../../checkpoints")
    pretrained_pred_head_checkpoint_paths = [
        f"{relative_checkpoint_path}/prediction_heads/dpt_feature/dust3r_512_dpt_head1_dpt.pth",
        f"{relative_checkpoint_path}/prediction_heads/dpt_feature/dust3r_512_dpt_head2_dpt.pth",
    ]
    pretrained_pred_head_regressor_checkpoint_paths = [
        f"{relative_checkpoint_path}/prediction_heads/dpt_reg_processor/dust3r_512_dpt_head1_reg_processor.pth",
        f"{relative_checkpoint_path}/prediction_heads/dpt_reg_processor/dust3r_512_dpt_head2_reg_processor.pth",
    ]
    # Initialize DUSt3R 512 DPT model using UniCeption modules
    dust3r_model = DUSt3R(
        name="dust3r_512_dpt",
        img_size=(512, 512),
        patch_embed_cls="ManyAR_PatchEmbed",
        pred_head_type="dpt",
        pretrained_encoder_checkpoint_path=f"{relative_checkpoint_path}/encoders/CroCo_Encoder_512_DUSt3R_dpt.pth",
        pretrained_decoder_checkpoint_path=f"{relative_checkpoint_path}/info_sharing/cross_attn_transformer/Two_View_Cross_Attention_Transformer_DUSt3R_512_dpt.pth",
        pretrained_pred_head_checkpoint_paths=pretrained_pred_head_checkpoint_paths,
        pretrained_pred_head_regressor_checkpoint_paths=pretrained_pred_head_regressor_checkpoint_paths,
    )
    print("DUSt3R model initialized successfully!")
    dust3r_model.cuda()

    import numpy as np
    import torch

    import requests
    from PIL import Image
    from io import BytesIO

    # Dual Image Encoder
    img0_url = (
        "https://raw.githubusercontent.com/naver/croco/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/assets/Chateau1.png"
    )
    img1_url = (
        "https://raw.githubusercontent.com/naver/croco/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/assets/Chateau2.png"
    )

    response = requests.get(img0_url)
    img0 = Image.open(BytesIO(response.content))

    response = requests.get(img1_url)
    img1 = Image.open(BytesIO(response.content))

    img0_tensor = torch.from_numpy(np.array(img0))[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255
    img1_tensor = torch.from_numpy(np.array(img1))[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255

    # normalize according to dust3r norm
    img0_tensor = (img0_tensor - 0.5) / 0.5
    img1_tensor = (img1_tensor - 0.5) / 0.5
    img_tensor = torch.cat((img0_tensor, img1_tensor), dim=0).cuda()

    # Run a dummy forward pass
    view1 = {"img": img_tensor, "instance": [0, 1], "data_norm_type": "dust3r"}
    view2 = {"img": view1["img"][[1, 0]].clone().cuda(), "instance": [1, 0], "data_norm_type": "dust3r"}

    res1, res2 = dust3r_model(view1, view2)
    print("Forward pass completed successfully!")

    points1 = res1["pts3d"].value[0].detach().cpu().numpy()
    points2 = res2["pts3d"].value[0].detach().cpu().numpy()

    # visualize the poitns in open3d
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points1.reshape(3, -1).T)
    pcd.colors = o3d.utility.Vector3dVector(np.array(img0)[..., :3].reshape(-1, 3) / 255)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2.reshape(3, -1).T)
    pcd2.colors = o3d.utility.Vector3dVector(np.array(img1)[..., :3].reshape(-1, 3) / 255)

    pcd += pcd2

    o3d.visualization.draw_geometries([pcd])

