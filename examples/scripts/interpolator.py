"""
Simple Model Interpolator

The script takes two checkpoints of the same model and outputs a merged checkpoint with linear interpolation.

Example usage:
    python interpolator.py \\
        --model-path ibm-granite/granite-3.3-8b-instruct \\
        --trained-model-path /path/to/checkpoint
"""
# Standard
import argparse

# Third Party
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def interpolate_models(
    model_path: str,
    trained_model_path: str,
    trained_model_weight: float = 0.5,
    output_model_path: str | None = None,
    torch_dtype: str | torch.dtype | None = "bfloat16",
) -> str:
    if output_model_path is None:
        output_model_path = f"{trained_model_path}_interp"

    if not (0.0 <= trained_model_weight <= 1.0):
        raise ValueError(f"trained_model_weight must be in [0,1], got {trained_model_weight}")

    model_kwargs = {}
    if torch_dtype is not None:
        if isinstance(torch_dtype, str):
            _torch_dtype = torch_dtype.lower()
            if _torch_dtype == "auto":
                model_kwargs["torch_dtype"] = "auto"
            else:
                _map = {
                    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                    "float16": torch.float16, "fp16": torch.float16,
                    "float32": torch.float32, "fp32": torch.float32,
                }
                if _torch_dtype not in _map:
                    raise ValueError(f"Unsupported --torch-dtype: {torch_dtype}")
                model_kwargs["torch_dtype"] = _map[_torch_dtype]
        else:
            model_kwargs["torch_dtype"] = torch_dtype

    # load original model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    state_dict = model.state_dict()
    original_model_weight = 1 - trained_model_weight
    for key in state_dict.keys():
        state_dict[key] = state_dict[key] * original_model_weight

    # load trained model
    trained_model = AutoModelForCausalLM.from_pretrained(
        trained_model_path,
        **model_kwargs,
    )
    trained_state_dict = trained_model.state_dict()
    for key in state_dict.keys():
        state_dict[key] += trained_state_dict[key] * trained_model_weight

    # save interpolated model
    model.save_pretrained(output_model_path, state_dict=state_dict)

    # copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_model_path)

    return output_model_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the original model",
    )
    parser.add_argument(
        "--trained-model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--trained-model-weight",
        type=float,
        default=0.5,
        help="Weight for the trained model",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        default=None,
        help="Path to the output model",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    model_path: str = args.model_path
    trained_model_path: str = args.trained_model_path
    trained_model_weight: float = args.trained_model_weight
    output_model_path: str | None = args.output_model_path
    torch_dtype: str | None = args.torch_dtype

    interpolate_models(
        model_path,
        trained_model_path,
        trained_model_weight=trained_model_weight,
        output_model_path=output_model_path,
        torch_dtype=torch_dtype,
    )


if __name__ == "__main__":
    main()
