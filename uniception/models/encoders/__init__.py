from uniception.models.encoders.base import UniCeptionEncoderBase, UniCeptionViTEncoderBase, IntermediateFeatureReturner


def _make_encoder(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:

    if encoder_str == "croco":
        raise NotImplementedError("CroCo encoder is not implemented yet.")
    elif encoder_str == "dummy":
        return UniCeptionEncoderBase(name="dummy", data_norm_type="dummy")


__all__ = ["_make_encoder", "UniCeptionEncoderBase", "UniCeptionViTEncoderBase", "IntermediateFeatureReturner"]
