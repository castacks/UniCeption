import argparse
import os

import torch
from uniception.models.factory import MatchAnythingModel
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformer,
    MultiViewCrossAttentionTransformerIFR,
)
from uniception.utils.profile import benchmark_torch_function, benchmark_torch_function_with_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile the MatchAnything model")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[2], help="batch sizes to profile")
    args = parser.parse_args()

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
    shape1, shape2 = img_size, img_size

    for batch_size in args.batch_sizes:
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
                execution_time, enoder_output = benchmark_torch_function_with_result(
                    model._encode_symmetrized, view1, view2
                )
                feat1, feat2 = enoder_output
                print(
                    f"\033[92mEncoding Symmetrized for batch size : {batch_size} completed in {execution_time:.3f} milliseconds\033[0m"
                )

                # Pass the features through the decoder
                decoder_input = MultiViewTransformerInput(features=[feat1, feat2])
                if model.head_type == "dpt":
                    execution_time, decoder_output = benchmark_torch_function_with_result(
                        model.info_sharing, decoder_input
                    )
                    final_decoder_multi_view_feat, intermediate_decoder_multi_view_feat = decoder_output
                elif model.head_type == "linear":
                    execution_time, final_decoder_multi_view_feat = benchmark_torch_function_with_result(
                        model.info_sharing, decoder_input
                    )
                print(
                    f"\033[92mDecoder for batch size : {batch_size} completed in {execution_time:.3f} milliseconds\033[0m"
                )

                # collect decoder features for the prediction heads
                if model.head_type == "dpt":
                    decoder_outputs = {
                        "1": [
                            feat1.float(),
                            intermediate_decoder_multi_view_feat[0].features[0].float(),
                            intermediate_decoder_multi_view_feat[1].features[0].float(),
                            final_decoder_multi_view_feat.features[0].float(),
                        ],
                        "2": [
                            feat2.float(),
                            intermediate_decoder_multi_view_feat[0].features[1].float(),
                            intermediate_decoder_multi_view_feat[1].features[1].float(),
                            final_decoder_multi_view_feat.features[1].float(),
                        ],
                    }
                elif model.head_type == "linear":
                    decoder_outputs = {
                        "1": final_decoder_multi_view_feat.features[0].float(),
                        "2": final_decoder_multi_view_feat.features[1].float(),
                    }

                # The prediction need precision, so we disable any autocasting here
                with torch.autocast("cuda", enabled=False):
                    # run the collected decoder features through the prediction heads
                    if model.info_sharing_and_head_structure == "dual+single":
                        # pass through head1 only and return the output
                        execution_time, head_output1 = benchmark_torch_function_with_result(
                            model._downstream_head, 1, decoder_outputs, shape1
                        )

                    elif model.info_sharing_and_head_structure in ["dual+dual", "dual+share"]:
                        # pass through head1 and head2 and return the output
                        execution_time_head_1, head_output1 = benchmark_torch_function_with_result(
                            model._downstream_head, 1, decoder_outputs, shape1
                        )
                        execution_time_head_2, head_output2 = benchmark_torch_function_with_result(
                            model._downstream_head, 2, decoder_outputs, shape1
                        )
                        execution_time = execution_time_head_1 + execution_time_head_2

                print(
                    f"\033[92mPrediction heads for batch size : {batch_size} completed in {execution_time:.3f} milliseconds\033[0m"
                )
