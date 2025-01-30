"""
Construct Match-Anything Model from Uniception Library
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from uniception.models.encoders import ViTEncoderInput, encoder_factory
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformer,
    MultiViewCrossAttentionTransformerIFR,
)
from uniception.models.libs.croco.pos_embed import RoPE2D
from uniception.models.prediction_heads.adaptors import (
    Covariance2DAdaptor,
    FlowAdaptor,
    FlowWithConfidenceAdaptor,
    MaskAdaptor,
)
from uniception.models.prediction_heads.base import AdaptorMap, PredictionHeadInput, PredictionHeadLayeredInput
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.linear import LinearFeature


# dust3r data structure for reducing passing duplicate images through the encoder
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


CLASSNAME_TO_ADAPTOR_CLASS = {
    "FlowWithConfidenceAdaptor": FlowWithConfidenceAdaptor,
    "FlowAdaptor": FlowAdaptor,
    "MaskAdaptor": MaskAdaptor,
}


class MatchAnythingModel(nn.Module):
    """
    Match-Anything Model from Uniception Library
    """

    def __init__(
        self,
        # Encoder configurations
        encoder_str: str,
        encoder_kwargs: Dict[str, Any] = {},
        img_size: Tuple[int, int] = (512, 512),
        # Info sharing & output head structure configurations
        info_sharing_and_head_structure: str = "dual+single",
        # Information sharing configurations
        input_embed_dim: int = 1024,
        transformer_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        position_encoding: str = "RoPE100",
        normalize_intermediate: bool = True,
        returned_intermediate_layers: Optional[List[int]] = None,
        info_sharing_kwargs: Dict[str, Any] = {},
        info_sharing_checkpoint_path: Optional[str] = None,
        # Prediction Heads & Adaptors
        head_type: str = "dpt",
        feature_head_kwargs: Dict[str, Any] = {},
        adaptors_kwargs: Dict[str, Any] = {},
        # Load Pretrained Weights
        pretrained_checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the Match-Anything Model from Uniception Library

        Args:
        - encoder_str (str): Encoder str name
        - encoder_kwargs (Dict[str, Any]): Encoder configurations
        - img_size (Tuple[int, int]): Image size

        - info_sharing_and_head_structure (str): Info Sharing & Head structure configurations
            - "share+share": two branch of the info_sharing share the same weights, and
              the two output heads share the same weights.
            - "dual+single": (default) two branch of the info_sharing have separate weights, and
              only one output head is used. It will output forward predictions.
            - "dual+dual": two branch of the info_sharing have separate weights, and
              two separate output heads are used. It will output forward and backward predictions.
            - "dual+share": two branch of the info_sharing have separate weights, and
              the two output heads share the same weights. It will output forward and backward predictions.

        - input_embed_dim (int): Input embedding dimension
        - transformer_dim (int): Transformer dimension
        - num_heads (int): Number of heads
        - num_layers (int): Number of layers
        - mlp_ratio (float): MLP ratio
        - qkv_bias (bool): whether to include bias in qkv projection in the transformer
        - qk_norm (bool): whether to normalize the query and key after linear projection
        - position_encoding (str): Position encoding method
        - normalize_intermediate (bool): When using DPT head, whether to apply layernorm
          to the returned intermediate features.
        - returned_intermediate_layers (Optional[List[int]]): When using DPT head, which
          layers to return intermediate features from.
        - info_sharing_checkpoint_path (Optional[str]): Path to the info_sharing checkpoint

        - head_type (str): Head type
            - "dpt": DPT head
            - "linear": Linear head
        - feature_head_kwargs (Dict[str, Any]): Feature head configurations
        - adaptors_kwargs (Dict[str, Any]): Adaptors configurations

        - pretrained_checkpoint_path (Optional[str]): Path to the pretrained checkpoint
        """

        super().__init__(*args, **kwargs)

        # initialize attributes
        self.encoder_str = encoder_str
        self.encoder_kwargs = encoder_kwargs
        self.img_size = img_size
        self.info_sharing_and_head_structure = info_sharing_and_head_structure
        self.input_embed_dim = input_embed_dim
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.position_encoding = position_encoding
        self.normalize_intermediate = normalize_intermediate
        self.returned_intermediate_layers = returned_intermediate_layers
        self.info_sharing_checkpoint_path = info_sharing_checkpoint_path
        self.head_type = head_type
        self.info_sharing_kwargs = info_sharing_kwargs
        self.feature_head_kwargs = feature_head_kwargs
        self.adaptors_kwargs = adaptors_kwargs
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        # initialize encoder
        self.encoder = encoder_factory(encoder_str, **encoder_kwargs)

        # initialize information sharing modules
        if position_encoding is None:
            self.pos_enc = None
        elif position_encoding.startswith("RoPE"):
            if position_encoding == "RoPE100":
                self.pos_enc = RoPE2D(freq=100, F0=1)
        else:
            raise ValueError(f"Position encoding method {position_encoding} not supported.")

        self.info_sharing: nn.Module

        if self.info_sharing_and_head_structure in ["dual+single", "dual+dual", "dual+share"]:
            if head_type == "dpt":
                self.info_sharing = MultiViewCrossAttentionTransformerIFR(
                    name="info_sharing",
                    input_embed_dim=self.input_embed_dim,
                    num_views=2,
                    depth=self.num_layers,
                    dim=self.transformer_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    custom_positional_encoding=self.pos_enc,
                    pretrained_checkpoint_path=self.info_sharing_checkpoint_path,
                    indices=self.returned_intermediate_layers,
                    norm_intermediate=self.normalize_intermediate,
                    **self.info_sharing_kwargs,
                )
            elif head_type == "linear":
                self.info_sharing = MultiViewCrossAttentionTransformer(
                    name="info_sharing",
                    input_embed_dim=self.input_embed_dim,
                    num_views=2,
                    depth=self.num_layers,
                    dim=self.transformer_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    custom_positional_encoding=self.pos_enc,
                    pretrained_checkpoint_path=self.info_sharing_checkpoint_path,
                    **self.info_sharing_kwargs,
                )
            else:
                raise ValueError(f"Head type {head_type} not supported.")
        else:
            raise ValueError(f"Info Sharing structure {info_sharing_and_head_structure} not supported.")

        # initialize prediction heads and adaptors
        if info_sharing_and_head_structure == "dual+single":
            self.add_module("head1", self._initialize_prediction_heads(head_type, feature_head_kwargs, adaptors_kwargs))
        elif decoder_structure == "dual+dual":
            self.add_module("head1", self._initialize_prediction_heads(head_type, feature_head_kwargs, adaptors_kwargs))
            self.add_module("head2", self._initialize_prediction_heads(head_type, feature_head_kwargs, adaptors_kwargs))
        elif info_sharing_and_head_structure == "dual+share":
            self.add_module("head1", self._initialize_prediction_heads(head_type, feature_head_kwargs, adaptors_kwargs))
            self.head2 = self.head1

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, strict=True, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
            model = cls(**ckpt["model_args"])
            model.load_state_dict(ckpt["model"], strict=strict)
            return model
        else:
            raise ValueError(f"Pretrained model {pretrained_model_name_or_path} not found.")

    def _initialize_prediction_heads(
        self, head_type: str, feature_head_kwargs: Dict[str, Any], adaptors_kwargs: Dict[str, Any]
    ):
        """
        Initialize prediction heads and adaptors

        Args:
        - head_type (str): Head type, either "dpt" or "linear"
        - feature_head_kwargs (Dict[str, Any]): Feature head configurations
        - adaptors_kwargs (Dict[str, Any]): Adaptors configurations

        Returns:
        - nn.Module: output head + adaptors
        """
        feature_processor: nn.Module
        if head_type == "dpt":
            feature_processor = nn.Sequential(
                DPTFeature(**feature_head_kwargs["dpt_feature"]),
                DPTRegressionProcessor(**feature_head_kwargs["dpt_processor"]),
            )
        elif head_type == "linear":
            feature_processor = LinearFeature(**feature_head_kwargs)
        else:
            raise ValueError(f"Head type {head_type} not supported.")

        adaptors = self._initialize_adaptors(adaptors_kwargs)

        return nn.Sequential(feature_processor, AdaptorMap(*adaptors.values()))

    def _initialize_adaptors(self, adaptors_kwargs: Dict[str, Any]):
        """
        Initialize a dict of adaptors

        Args:
        - adaptors_kwargs (Dict[str, Any]): Adaptors configurations

        Returns:
        - Dict[str, nn.Module]: dict of adaptors, from adaptor's name to the adaptor
        """
        return {
            name: CLASSNAME_TO_ADAPTOR_CLASS[configs["class"]](**configs["kwargs"])
            for name, configs in adaptors_kwargs.items()
        }

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

    @torch.compiler.disable(recursive=False)
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
        if self.head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.head_type == "dpt":
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)

    def forward(self, view1, view2):
        """
        Forward pass of the Match-Anything Model
        1. Encodes two input images (view1 and view2) into feature embeddings
        2. Passes the embeddings through the info_sharing
        3. Passes the info_sharing output through the prediction heads and adaptors

        Args:
        - view1 (Dict[str, Any]): Input view 1
          - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
          - instance (List[int]): List of instance indices, or id of the input image
          - data_norm_type (str): Data normalization type, see uniception.models.encoders.IMAGE_NORMALIZATION_DICT
        - view2 (Dict[str, Any]): Input view 2
          - (same structure as view1)

        Returns:
        - Dict[str, Any]: Output results
          - flow (Dict[str, torch.Tensor]): Flow output
            - flow_output (torch.Tensor): Flow output tensor, BCHW
            - flow_output_conf (torch.Tensor): Flow output confidence tensor, BCHW
          - occlusion (Dict[str, torch.Tensor]): Occlusion output
            - mask (torch.Tensor): probibility of not occluded, BCHW tensor
            - logits (torch.Tensor): logits of the mask before sigmoid, BCHW
        """

        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1, feat2 = self._encode_symmetrized(view1, view2)

        # Pass the features through the decoder
        decoder_input = MultiViewTransformerInput(features=[feat1, feat2])
        if self.head_type == "dpt":
            final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
                info_sharing_input
            )
        elif self.head_type == "linear":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)

        # collect info_sharing features for the prediction heads
        if self.head_type == "dpt":
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
        elif self.head_type == "linear":
            info_sharing_outputs = {
                "1": final_info_sharing_multi_view_feat.features[0].float(),
                "2": final_info_sharing_multi_view_feat.features[1].float(),
            }

        # The prediction need precision, so we disable any autocasting here
        with torch.autocast("cuda", enabled=False):
            # run the collected decoder features through the prediction heads
            if self.decoder_structure == "dual+single":
                # pass through head1 only and return the output
                head_output1 = self._downstream_head(1, decoder_outputs, shape1)

                res1 = {
                    "flow": head_output1["flow_with_confidence"].value,
                    "flow_conf": head_output1["flow_with_confidence"].confidence,
                    "non_occluded_fwd": head_output1["non_occluded_mask"],
                }

                return {
                    "flow": {
                        "flow_output": res1["flow"],
                        "flow_output_conf": res1["flow_conf"],
                    },
                    "occlusion": {
                        "mask": res1["non_occluded_fwd"].mask,
                        "logits": res1["non_occluded_fwd"].logits,
                    },
                }
            elif self.info_sharing_and_head_structure in ["dual+dual", "dual+share"]:
                # pass through head1 and head2 and return the output
                head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)
                head_output2 = self._downstream_head(2, info_sharing_outputs, shape1)

                if "flow" in head_output1 and "flow_cov" in head_output1:
                    raise NotImplementedError("Flow with covariance not implemented for dual+share or dual+dual")

                res1 = {
                    "flow": head_output1["flow_with_confidence"].value,
                    "flow_conf": head_output1["flow_with_confidence"].confidence,
                    "non_occluded_fwd": head_output1["non_occluded_mask"],
                }

                res2 = {
                    "flow": head_output2["flow_with_confidence"].value,
                    "flow_conf": head_output2["flow_with_confidence"].confidence,
                    "non_occluded_bwd": head_output2["non_occluded_mask"],
                }

                result = {}
                result["flow"] = {
                    "flow_output": torch.cat((res1["flow"], res2["flow"]), dim=0),
                    "flow_output_conf": torch.cat((res1["flow_conf"], res2["flow_conf"]), dim=0),
                }

                result["occlusion"] = {
                    "mask": torch.cat((res1["non_occluded_fwd"].mask, res2["non_occluded_bwd"].mask)),
                    "logits": torch.cat((res1["non_occluded_fwd"].logits, res2["non_occluded_bwd"].logits)),
                }

                return result
