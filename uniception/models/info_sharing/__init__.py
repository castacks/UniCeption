from uniception.models.info_sharing.alternating_attention_transformer import (
    MultiViewAlternatingAttentionTransformer,
    MultiViewAlternatingAttentionTransformerIFR,
)
from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformer,
    MultiViewCrossAttentionTransformerIFR,
    MultiViewTransformerInput,
)
from uniception.models.info_sharing.diff_cross_attention_transformer import (
    DifferentialMultiViewCrossAttentionTransformer,
    DifferentialMultiViewCrossAttentionTransformerIFR,
)
from uniception.models.info_sharing.global_attention_transformer import (
    MultiViewGlobalAttentionTransformer,
    MultiViewGlobalAttentionTransformerIFR,
    MultiViewGlobalAttentionTransformerFFIFR
)

from uniception.models.info_sharing.base import MultiViewTransformerFeedForwardInput

INFO_SHARING_CLASSES = {
    "cross_attention": (MultiViewCrossAttentionTransformer, MultiViewCrossAttentionTransformerIFR),
    "diff_cross_attention": (
        DifferentialMultiViewCrossAttentionTransformer,
        DifferentialMultiViewCrossAttentionTransformerIFR,
    ),
    "alternating_attention": (
        MultiViewAlternatingAttentionTransformer,
        MultiViewAlternatingAttentionTransformerIFR,
    ),
    "global_attention": (MultiViewGlobalAttentionTransformer, MultiViewGlobalAttentionTransformerIFR, MultiViewGlobalAttentionTransformerFFIFR),
}

__all__ = ["INFO_SHARING_CLASSES", "MultiViewTransformerInput", "MultiViewTransformerFeedForwardInput"]
