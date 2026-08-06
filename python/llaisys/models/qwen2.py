from ctypes import byref, c_int, c_void_p, c_int64
from pathlib import Path
from typing import Sequence
from ..libllaisys import (
    LIB_LLAISYS,
    DataType,
    DeviceType,
    LlaisysQwen2Meta,
)

import safetensors
import ml_dtypes
import json


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config = json.loads(
            (model_path / "config.json").read_text()
        )

        hs = config["hidden_size"]
        nh = config["num_attention_heads"]

        meta = LlaisysQwen2Meta(
            DataType.BF16,
            config["num_hidden_layers"],
            hs,
            nh,
            config["num_key_value_heads"],
            config.get("head_dim", hs // nh),
            config["intermediate_size"],
            config["max_position_embeddings"],
            config["vocab_size"],
            config["rms_norm_eps"],
            config.get("rope_theta", 10000.0),
            config["eos_token_id"],
        )

        self.end_token = config["eos_token_id"]
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), device, device_ids, 1
        )

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(
                str(file), framework="numpy", device="cpu"
            ) as tensors:
                for name in tensors.keys():
                    data = tensors.get_tensor(name)
                    LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                        self._model,
                        name.encode(),
                        c_void_p(data.ctypes.data),
                        data.nbytes,
                    )

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        outputs = [int(token) for token in inputs]

        if not outputs or max_new_tokens <= 0:
            return outputs

        capacity = len(outputs) + max_new_tokens

        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model, capacity)

        pending = outputs

        for _ in range(max_new_tokens):
            token_ids = (c_int64 * len(pending))(*pending)

            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    token_ids,
                    len(pending)
                )
            )

            outputs.append(next_token)

            if next_token == self.end_token:
                break

            pending = [next_token]

        return outputs