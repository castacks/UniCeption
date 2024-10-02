from typing import List, Tuple

import torch
import torch.nn as nn
from uniception.models.encoders import ViTEncoderInput
from uniception.models.encoders.croco import CroCoEncoder
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformer,
    MultiViewCrossAttentionTransformerIFR,
    MultiViewCrossAttentionTransformerInput,
)
from uniception.models.libs.croco.pos_embed import RoPE2D, get_2d_sincos_pos_embed
from uniception.models.prediction_heads.adaptors import PointMapWithConfidenceAdaptor
from uniception.models.prediction_heads.base import AdaptorInput, PredictionHeadInput, PredictionHeadLayeredInput
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.linear import LinearFeature


def is_symmetrized(gt1, gt2):
    "Function to check if input pairs are symmetrized, i.e., (a, b) and (b, a) always exist in the input"
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def interleave(tensor1, tensor2):
    "Interleave two tensors along the first dimension (used to avoid redundant encoding for symmetrized pairs)"
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
        pretrained_checkpoint_path: str = None,
        pretrained_encoder_checkpoint_path: str = None,
        pretrained_info_sharing_checkpoint_path: str = None,
        pretrained_pred_head_checkpoint_paths: List[str] = [None, None],
        pretrained_pred_head_regressor_checkpoint_paths: List[str] = [None, None],
        override_encoder_checkpoint_attributes: bool = False,
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
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            pretrained_encoder_checkpoint_path (str): Path to pretrained encoder checkpoint. (default: None)
            pretrained_info_sharing_checkpoint_path (str): Path to pretrained info_sharing checkpoint. (default: None)
            pretrained_pred_head_checkpoint_paths (List[str]): Paths to pretrained prediction head checkpoints. (default: None)
            pretrained_pred_head_regressor_checkpoint_paths (List[str]): Paths to pretrained prediction head regressor checkpoints. (default: None)
            override_encoder_checkpoint_attributes (bool): Whether to override encoder checkpoint attributes. (default: False)
        """
        super().__init__(*args, **kwargs)

        # Initalize the attributes
        self.name = name
        self.data_norm_type = data_norm_type
        self.img_size = img_size
        self.patch_embed_cls = patch_embed_cls
        self.pred_head_type = pred_head_type
        self.pred_head_output_dim = pred_head_output_dim
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pos_embed = pos_embed
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.pretrained_encoder_checkpoint_path = pretrained_encoder_checkpoint_path
        self.pretrained_info_sharing_checkpoint_path = pretrained_info_sharing_checkpoint_path
        self.pretrained_pred_head_checkpoint_paths = pretrained_pred_head_checkpoint_paths
        self.pretrained_pred_head_regressor_checkpoint_paths = pretrained_pred_head_regressor_checkpoint_paths
        self.override_encoder_checkpoint_attributes = override_encoder_checkpoint_attributes

        # Initialize RoPE for the CroCo Encoder & Two-View Cross Attention Transformer
        freq = float(pos_embed[len("RoPE") :])
        self.rope = RoPE2D(freq=freq)

        # Initialize Encoder
        self.encoder = CroCoEncoder(
            name=name,
            data_norm_type=data_norm_type,
            patch_embed_cls=patch_embed_cls,
            img_size=img_size,
            pretrained_checkpoint_path=pretrained_encoder_checkpoint_path,
            override_checkpoint_attributes=override_encoder_checkpoint_attributes,
        )

        # Initialize Multi-View Cross Attention Transformer
        if self.pred_head_type == "linear":
            # Returns only normalized last layer features
            self.info_sharing = MultiViewCrossAttentionTransformer(
                name="base_info_sharing",
                input_embed_dim=self.encoder.enc_embed_dim,
                num_views=2,
                custom_positional_encoding=self.rope,
                pretrained_checkpoint_path=pretrained_info_sharing_checkpoint_path,
            )
        elif self.pred_head_type == "dpt":
            # Returns intermediate features and normalized last layer features
            self.info_sharing = MultiViewCrossAttentionTransformerIFR(
                name="base_info_sharing",
                input_embed_dim=self.encoder.enc_embed_dim,
                num_views=2,
                indices=[5, 8],
                norm_intermediate=False,
                custom_positional_encoding=self.rope,
                pretrained_checkpoint_path=pretrained_info_sharing_checkpoint_path,
            )
        else:
            raise ValueError(f"Invalid prediction head type: {pred_head_type}. Must be 'linear' or 'dpt'.")

        # Initialize Prediction Heads
        if pred_head_type == "linear":
            # Initialize Prediction Head 1
            self.head1 = LinearFeature(
                input_feature_dim=self.info_sharing.dim,
                output_dim=pred_head_output_dim,
                patch_size=self.encoder.patch_size,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[0],
            )
            # Initialize Prediction Head 2
            self.head2 = LinearFeature(
                input_feature_dim=self.info_sharing.dim,
                output_dim=pred_head_output_dim,
                patch_size=self.encoder.patch_size,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[1],
            )
        elif pred_head_type == "dpt":
            # Initialze Predction Head 1
            self.dpt_feature_head1 = DPTFeature(
                patch_size=self.encoder.patch_size,
                hooks=[0, 1, 2, 3],
                input_feature_dims=[self.encoder.enc_embed_dim] + [self.info_sharing.dim] * 3,
                feature_dim=pred_head_feature_dim,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[0],
            )
            self.dpt_regressor_head1 = DPTRegressionProcessor(
                input_feature_dim=pred_head_feature_dim,
                output_dim=pred_head_output_dim,
                pretrained_checkpoint_path=pretrained_pred_head_regressor_checkpoint_paths[0],
            )
            self.head1 = nn.Sequential(self.dpt_feature_head1, self.dpt_regressor_head1)
            # Initialize Prediction Head 2
            self.dpt_feature_head2 = DPTFeature(
                patch_size=self.encoder.patch_size,
                hooks=[0, 1, 2, 3],
                input_feature_dims=[self.encoder.enc_embed_dim] + [self.info_sharing.dim] * 3,
                feature_dim=pred_head_feature_dim,
                pretrained_checkpoint_path=pretrained_pred_head_checkpoint_paths[1],
            )
            self.dpt_regressor_head2 = DPTRegressionProcessor(
                input_feature_dim=pred_head_feature_dim,
                output_dim=pred_head_output_dim,
                pretrained_checkpoint_path=pretrained_pred_head_regressor_checkpoint_paths[1],
            )
            self.head2 = nn.Sequential(self.dpt_feature_head2, self.dpt_regressor_head2)

        # Initialize Final Output Adaptor
        self.adaptor = PointMapWithConfidenceAdaptor(
            name="pointmap",
            pointmap_mode=depth_mode[0],
            pointmap_vmin=depth_mode[1],
            pointmap_vmax=depth_mode[2],
            confidence_type=conf_mode[0],
            confidence_vmin=conf_mode[1],
            confidence_vmax=conf_mode[2],
        )

        # Load pretrained weights
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DUSt3R weights from {self.pretrained_checkpoint_path} ...")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def _encode_image_pairs(self, img1, img2, data_norm_type):
        "Encode two different batches of images (each batch can have different image shape)"
        if img1.shape[-2:] == img2.shape[-2:]:
            encoder_input = ViTEncoderInput(image=torch.cat((img1, img2), dim=0), data_norm_type=data_norm_type)
            encoder_output = self.encoder(encoder_input)
            out, out2 = encoder_output.features.chunk(2, dim=0)
        else:
            encoder_input = ViTEncoderInput(image=img1, data_norm_type=data_norm_type)
            out = self.encoder(encoder_input)
            out = out.features
            encoder_input2 = ViTEncoderInput(image=img2)
            out2 = self.encoder(encoder_input2)
            out2 = out2.features

        return out, out2

    def _encode_symmetrized(self, view1, view2):
        "Encode image pairs accounting for symmetrization, i.e., (a, b) and (b, a) always exist in the input"
        img1 = view1["img"]
        img2 = view2["img"]
        if is_symmetrized(view1, view2):
            # Computing half of forward pass'
            feat1, feat2 = self._encode_image_pairs(img1[::2], img2[::2], data_norm_type=view1["data_norm_type"])
            feat1, feat2 = interleave(feat1, feat2)
        else:
            feat1, feat2 = self._encode_image_pairs(img1, img2, data_norm_type=view1["data_norm_type"])

        return feat1, feat2

    def _downstream_head(self, head_num, decout, img_shape):
        "Run the respective prediction heads"
        head = getattr(self, f"head{head_num}")
        if self.pred_head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.pred_head_type == "dpt":
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)

    def forward(self, view1, view2):
        """
        Forward pass for DUSt3R performing the following operations:
        1. Encodes the two input views (images).
        2. Combines the encoded features using a two-view cross-attention transformer.
        3. Passes the combined features through the respective prediction heads.
        4. Returns the processed final outputs for both views.

        Args:
            view1 (dict): Dictionary containing the first view's images and instance information.
                          "img" is a required key and value is a tensor of shape (B, C, H, W).
            view2 (dict): Dictionary containing the second view's images and instance information.
                          "img" is a required key and value is a tensor of shape (B, C, H, W).

        Returns:
            Tuple[dict, dict]: A tuple containing the final outputs for both views.
        """
        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1, feat2 = self._encode_symmetrized(view1, view2)

        # Combine all images into view-centric representation
        info_sharing_input = MultiViewCrossAttentionTransformerInput(features=[feat1, feat2])
        if self.pred_head_type == "linear":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
        elif self.pred_head_type == "dpt":
            final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
                info_sharing_input
            )

        if self.pred_head_type == "linear":
            # Define feature dictionary for linear head
            info_sharing_outputs = {
                "1": final_info_sharing_multi_view_feat.features[0].float(),
                "2": final_info_sharing_multi_view_feat.features[1].float(),
            }
        elif self.pred_head_type == "dpt":
            # Define feature dictionary for DPT head
            info_sharing_outputs = {
                "1": [
                    feat1.float(),
                    intermediate_info_sharing_multi_view_feat[0].features[0].float(),
                    intermediate_info_sharing_multi_view_feat[1].features[0].float(),
                    final_info_sharing_multi_view_feat.features[0].float(),
                ],
                "2": [
                    feat2.float(),
                    intermediate_info_sharing_multi_view_feat[0].features[1].float(),
                    intermediate_info_sharing_multi_view_feat[1].features[1].float(),
                    final_info_sharing_multi_view_feat.features[1].float(),
                ],
            }

        # Downstream task prediction
        with torch.autocast("cuda", enabled=False):
            # Prediction heads
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)
            head_output2 = self._downstream_head(2, info_sharing_outputs, shape2)

            # Post-process outputs
            final_output1 = self.adaptor(
                AdaptorInput(adaptor_feature=head_output1.decoded_channels, output_shape_hw=shape1)
            )
            final_output2 = self.adaptor(
                AdaptorInput(adaptor_feature=head_output2.decoded_channels, output_shape_hw=shape2)
            )

            # Convert outputs to dictionary
            res1 = {
                "pts3d": final_output1.value.permute(0, 2, 3, 1).contiguous(),
                "conf": final_output1.confidence.permute(0, 2, 3, 1).contiguous(),
            }
            res2 = {
                "pts3d_in_other_view": final_output2.value.permute(0, 2, 3, 1).contiguous(),
                "conf": final_output2.confidence.permute(0, 2, 3, 1).contiguous(),
            }

        return res1, res2
