"""
Adaptors for the UniCeption Prediction Heads.
"""

from functools import lru_cache
from math import isfinite
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from uniception.models.prediction_heads import (
    AdaptorInput,
    MaskAdaptorOutput,
    RegressionAdaptorOutput,
    RegressionWithConfidenceAdaptorOutput,
    UniCeptionAdaptorBase,
)


class FlowAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        flow_mean: Union[Tuple[float, float], List[float]],
        flow_std: Union[Tuple[float, float], List[float]],
        base_shape: Tuple[int, int],
        scale_strategy: str,
        output_normalized_coordinate: bool = False,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Flow head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            flow_mean (torch.Tensor): (2,) Mean of the flow.
            flow_std (torch.Tensor): (2,) Standard deviation of the flow.
            base_shape (Tuple[int, int]): Base shape of the flow mean and std.
            scale_strategy (str): Strategy for scaling the flow, either
            - none: No scaling, network will be unnormalized with the given mean and std for all input shapes
            - scale_width: scale the output for "none" by actual width divided by base width for both X and Y
            - scale_height: scale the output for "none" by actual height divided by base height for both X and Y
            - scale_both: scale the output for "none" by actual dimension / base dimension individually for X and Y
            output_normalized_coordinate (bool): If True, will subtract the (X, Y) coordinate of the output pixel from input x after it is being scaled to pixel coordinates.
            In other words, the network will predict the pixel position that the source pixel will land on the target image, rather than the flow.
        """
        super().__init__(name, required_channels=2, *args, **kwargs)

        self.name: str = name

        flow_mean = list(flow_mean)
        flow_std = list(flow_std)

        # handle the case where flow_mean and flow_std are passed as tuples
        if isinstance(flow_mean, tuple) or isinstance(flow_mean, list):
            flow_mean = torch.tensor(flow_mean, dtype=torch.float32)
            assert flow_mean.shape == (2,), f"Flow mean must be a 2D tensor, got {flow_mean.shape}"

        if isinstance(flow_std, tuple) or isinstance(flow_std, list):
            flow_std = torch.tensor(flow_std, dtype=torch.float32)
            assert flow_std.shape == (2,), f"Flow std must be a 2D tensor, got {flow_std.shape}"

        self.register_buffer("flow_mean", flow_mean.view(1, 2, 1, 1))
        self.register_buffer("flow_std", flow_std.view(1, 2, 1, 1))

        self.base_shape = list(base_shape)
        self.scale_strategy = scale_strategy
        self.output_normalized_coordinate = output_normalized_coordinate

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the FlowAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor.

        Returns:
            AdaptorOutput: Output of the adaptor.
        """

        x = adaptor_input.adaptor_feature

        # check the number of channels to avoid passing BHWC features
        _, C, _, _ = x.shape
        assert C == 2, f"FlowAdaptor requires BCHW format with 2 channels, got {C} channels"

        output_shape = adaptor_input.output_shape_hw

        if not self.output_normalized_coordinate:

            x_scale, y_scale = self._get_xy_scale(output_shape)

            # scale the flow by stored mean, std and scaling factors
            flow_mean = self.flow_mean * torch.tensor([x_scale, y_scale], dtype=torch.float32, device=x.device).view(
                1, 2, 1, 1
            )
            flow_std = self.flow_std * torch.tensor([x_scale, y_scale], dtype=torch.float32, device=x.device).view(
                1, 2, 1, 1
            )

            # unnormalize the flow
            x = x * flow_std + flow_mean
        else:
            # optionally subtract the coordinate bias
            wh_normalizer = torch.tensor(
                adaptor_input.output_shape_hw[::-1], dtype=torch.float32, device=x.device
            ).view(1, 2, 1, 1)

            x = 0.5 * (x + 1) * wh_normalizer + 0.5

            coords = self._get_coordinate_bias(output_shape, x.device)
            x = x - coords

        return RegressionAdaptorOutput(value=x)

    def _get_xy_scale(self, output_shape: Tuple[int, int]):
        """
        Get the scaling factor for the X and Y dimensions.

        Args:
            output_shape (Tuple[int, int]): HW Shape of the output.

        Returns:
            Tuple[float, float]: Scaling factors for X and Y dimensions.
        """
        if self.scale_strategy == "none":
            return 1.0, 1.0
        elif self.scale_strategy == "scale_width":
            return output_shape[1] / self.base_shape[1], output_shape[1] / self.base_shape[1]
        elif self.scale_strategy == "scale_height":
            return output_shape[0] / self.base_shape[0], output_shape[0] / self.base_shape[0]
        elif self.scale_strategy == "scale_both":
            return output_shape[1] / self.base_shape[1], output_shape[0] / self.base_shape[0]
        else:
            raise ValueError(f"Invalid scaling strategy: {self.scale_strategy}")

    @lru_cache(maxsize=10)
    def _get_coordinate_bias(self, output_shape: Tuple[int, int], device: str):
        """
        Get the (X, Y) coordinate image for the given output shape.

        Args:
            output_shape (Tuple[int, int]): HW Shape of the output.
            device: device to store the tensor on

        Returns:
            torch.Tensor: (2, H, W) tensor with X and Y coordinates, at device. This coordinate value will
            include 0.5 px offset - i.e. the center of the top-left pixel is (0.5, 0.5).
        """

        H, W = output_shape

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, device=device, dtype=torch.float32) + 0.5,
                torch.arange(0, H, device=device, dtype=torch.float32) + 0.5,
                indexing="xy",
            ),
            dim=0,
        )

        return coords


class DepthAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = -np.inf, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for the Depth head in UniCeption.
        """
        super().__init__(name, required_channels=1, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
        assert self.no_bounds

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the DepthAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor.
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        x = adaptor_input.adaptor_feature

        # check the number of channels to avoid passing BHWC features
        _, C, _, _ = x.shape
        assert C == 1, f"DepthAdaptor requires BCHW format with 1 channels, got {C} channels"

        output_depth = None

        if self.mode == "linear":
            if self.no_bounds:
                output_depth = x
            else:
                output_depth = x.clip(self.vmin, self.vmax)
        elif self.mode == "square":
            if self.no_bounds:
                output_depth = x**2
            else:
                output_depth = (x**2).clip(self.vmin, self.vmax)
        elif self.mode == "exp":
            output_depth = torch.expm1(x)

        return RegressionAdaptorOutput(value=output_depth)


class PointMapAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = -np.inf, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for the Depth head in UniCeption.
        """
        super().__init__(name, required_channels=3, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
        assert self.no_bounds

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the PointMapAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor.
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        xyz = adaptor_input.adaptor_feature
        mode, vmin, vmax = self.mode, self.vmin, self.vmax

        no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
        assert no_bounds

        if mode == "linear":
            if no_bounds:
                return RegressionAdaptorOutput(value=xyz)  # [-inf, +inf]
            return RegressionAdaptorOutput(value=xyz.clip(min=vmin, max=vmax))

        # distance to origin
        d = xyz.norm(dim=1, keepdim=True)
        xyz = xyz / d.clip(min=1e-8)

        if mode == "square":
            return RegressionAdaptorOutput(value=xyz * d.square())

        if mode == "exp":
            return RegressionAdaptorOutput(value=xyz * torch.expm1(d))


class ConfidenceAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        confidence_type: str,
        vmin: float,
        vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Confidence head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            confidence_type (str): Type of the confidence, either
            - exp: Exponential confidence
            - sigmoid: Sigmoid confidence
            vmin (float): Minimum value of the confidence.
            vmax (float): Maximum value of the confidence.
        """
        super().__init__(name, required_channels=1, *args, **kwargs)

        self.confidence_type = confidence_type
        self.vmin = vmin
        self.vmax = vmax

        assert vmin < vmax, "vmin must be less than vmax"

        if confidence_type == "sigmoid":
            assert isfinite(vmin) and isfinite(vmax), "vmin and vmax must be finite for sigmoid confidence"
            assert vmin >= 0

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the ConfidenceAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor.
        Returns:
            AdaptorOutput: Output of the adaptor.
        """

        x = adaptor_input.adaptor_feature

        if self.confidence_type == "exp":
            confidence = self.vmin + x.exp().clip(max=self.vmax - self.vmin)

            return RegressionAdaptorOutput(value=confidence)

        elif self.confidence_type == "sigmoid":
            confidence = torch.sigmoid(x)

            confidence = confidence * (self.vmax - self.vmin) + self.vmin

            return RegressionAdaptorOutput(value=confidence)


class MaskAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Mask head in UniCeption.
        """
        super().__init__(name, required_channels=1, *args, **kwargs)

    def forward(self, adaptor_input: AdaptorInput):
        x = adaptor_input.adaptor_feature

        mask = torch.sigmoid(x)

        return MaskAdaptorOutput(logits=x, mask=mask)


class ValueWithConfidenceAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        value_adaptor: UniCeptionAdaptorBase,
        confidence_adaptor: UniCeptionAdaptorBase,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Value with Confidence head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            value_adaptor (UniCeptionAdaptorBase): Adaptor for the value.
            confidence_adaptor (UniCeptionAdaptorBase): Adaptor for the confidence.
        """

        super().__init__(
            name,
            required_channels=value_adaptor.required_channels + confidence_adaptor.required_channels,
            *args,
            **kwargs,
        )

        self.value_adaptor = value_adaptor
        self.confidence_adaptor = confidence_adaptor

    def forward(self, adaptor_input: AdaptorInput):
        value_input, confidence_input = torch.split(
            adaptor_input.adaptor_feature,
            [self.value_adaptor.required_channels, self.confidence_adaptor.required_channels],
            dim=1,
        )
        value_adaptor_input = AdaptorInput(adaptor_feature=value_input, output_shape_hw=adaptor_input.output_shape_hw)
        confidence_adaptor_input = AdaptorInput(
            adaptor_feature=confidence_input, output_shape_hw=adaptor_input.output_shape_hw
        )
        value_output = self.value_adaptor(value_adaptor_input)
        confidence_output = self.confidence_adaptor(confidence_adaptor_input)

        return RegressionWithConfidenceAdaptorOutput(value=value_output.value, confidence=confidence_output.value)


class FlowWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # flow adaptor
        flow_mean: torch.Tensor,
        flow_std: torch.Tensor,
        base_shape: Tuple[int, int],
        scale_strategy: str,
        output_normalized_coordinate: bool,
        # confidence adaptor
        confidence_type: str,
        vmin: float,
        vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Flow with Confidence head in UniCeption.
        """
        flow_adaptor = FlowAdaptor(
            name=f"{name}",
            flow_mean=flow_mean,
            flow_std=flow_std,
            base_shape=base_shape,
            scale_strategy=scale_strategy,
            output_normalized_coordinate=output_normalized_coordinate,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=vmin, vmax=vmax
        )

        super().__init__(name, value_adaptor=flow_adaptor, confidence_adaptor=confidence_adaptor, *args, **kwargs)


class PointMapWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # pointmap adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        # confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap with Confidence head in UniCeption.
        """
        pointmap_adaptor = PointMapAdaptor(name=f"{name}", mode=pointmap_mode, vmin=pointmap_vmin, vmax=pointmap_vmax)

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(name, value_adaptor=pointmap_adaptor, confidence_adaptor=confidence_adaptor, *args, **kwargs)
