# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only T5Gemma 2 model.

T5Gemma 2 is an encoder-decoder model adapted from Gemma 3
(reference: `transformers/models/t5gemma2/modeling_t5gemma2.py`):

* The encoder is a bidirectional Gemma 3 text stack. Images go through a
  SigLIP vision tower and replace the image soft tokens of the encoder input.
* Every decoder layer uses *merged attention*: one softmax over the decoder's
  own (causal, possibly sliding-window) keys and all encoder outputs, with the
  same q/k/v/o projections for both parts.

vLLM V1 runs the encoder of an encoder-decoder model through the multimodal
encoder path (`embed_multimodal`). The processor therefore packs the whole
encoder prompt, i.e. its token IDs plus all of its images, into a single
multimodal item, so that text-only requests also reach the encoder.

Merged attention is computed as two partial attentions, one over the decoder
self-attention KV cache and one over the cross-attention KV cache. Merging
them by their log-sum-exp (LSE) is exactly one softmax over the union of both
key sets.

LoRA covers the text encoder and the decoder, but not the vision tower. The
text encoder runs on the batch of encoder tokens, whose adapter mapping vLLM
only builds for multimodal "tower" modules, so it is registered as the tower.
The decoder projects the encoder outputs into cross-attention K/V with that
same per-encoder-token mapping.
"""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Gemma3Processor
from transformers.models.t5gemma2 import (
    T5Gemma2Config,
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
)

from vllm.config import CacheConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.inputs import MultiModalDataDict, MultiModalEncDecInput, MultiModalInput
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    EncDecMultiModalProcessor,
    ProcessorInputs,
    TimingContext,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from .gemma3 import Gemma3MLP
from .gemma3_mm import (
    Gemma3MultiModalProcessor,
    Gemma3MultiModalProjector,
    Gemma3ProcessingInfo,
)
from .interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal
from .module_mapping import MultiModelKeys
from .siglip import SiglipVisionModel
from .utils import AutoWeightsLoader, WeightsMapper, extract_layer_index, maybe_prefix

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase

# The encoder input of one request (its token IDs and all of its images) is
# packed into one multimodal item of this modality, which users never pass
# themselves (see `T5Gemma2MultiModalProcessor._pack_encoder_input`). Keeping
# it apart from "image" lets `limit_mm_per_prompt={"image": 0}` skip the
# vision tower for text-only use.
_ENCODER_MODALITY = "text"
_ENCODER_INPUT_IDS = "encoder_input_ids"
_PIXEL_VALUES = "pixel_values"

_LORA_DECODER_PREFIX = "model.decoder"
_LORA_ENCODER_PREFIX = "model.encoder.text_model"


# --------------------------------------------------------------------------- #
# Multimodal processing
# --------------------------------------------------------------------------- #


class T5Gemma2ProcessingInfo(Gemma3ProcessingInfo):
    def get_hf_config(self) -> T5Gemma2Config:
        return self.ctx.get_hf_config(T5Gemma2Config)

    def get_hf_processor(self, **kwargs: object) -> Gemma3Processor:
        return self.ctx.get_hf_processor(Gemma3Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, _ENCODER_MODALITY: 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # Only the encoder item reaches the encoder; images are part of it.
        # Its size is bounded by the model length.
        return {_ENCODER_MODALITY: seq_len}

    def get_image_seq_len(self) -> int:
        """Number of encoder tokens of the largest image after expansion."""
        image_size = self.get_image_size_with_most_features()
        repl = self.get_image_repl(
            image_width=image_size.width,
            image_height=image_size.height,
            processor=self.get_hf_processor(),
            mm_kwargs={},
        )
        return len(repl.full)

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()
        image_processor = processor.image_processor
        max_num_crops = (
            image_processor.pan_and_scan_max_num_crops
            if image_processor.do_pan_and_scan
            else 1
        )
        native_size = self.get_hf_config().encoder.vision_config.image_size
        return ImageSize(height=native_size * max_num_crops, width=native_size)


class T5Gemma2DummyInputsBuilder(BaseDummyInputsBuilder[T5Gemma2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        boi_token = self.info.get_hf_processor().boi_token
        return boi_token * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class T5Gemma2MultiModalProcessor(
    EncDecMultiModalProcessor[T5Gemma2ProcessingInfo],
    Gemma3MultiModalProcessor,
):
    """Processor that turns the encoder prompt into one multimodal item.

    The image placeholders are expanded with the Gemma 3 rules (inherited from
    `Gemma3MultiModalProcessor`). Afterwards, the expanded encoder token IDs
    and the pixel values of every image are packed into a single item of
    `_ENCODER_MODALITY` whose placeholder covers the whole encoder prompt.
    The decoder prompt starts empty, so the renderer only adds the decoder
    start token.
    """

    # The encoder consumes the prompt tokens, so text-only prompts must still
    # go through `apply` to become an encoder item.
    process_text_only_encoder_prompt = True

    def create_encoder_prompt(
        self,
        prompt: list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        return prompt

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return []

    def get_dummy_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> ProcessorInputs:
        image_seq_len = self.info.get_image_seq_len()
        if "image" not in mm_counts and mm_counts.get(_ENCODER_MODALITY, 0):
            # Profiling the encoder item: the worst case holds as many images
            # as are allowed and fit into the encoder sequence.
            max_images = self.info.allowed_mm_limits.get("image", 0)
            mm_counts = {"image": min(max_images, seq_len // image_seq_len)}
        inputs = super().get_dummy_inputs(seq_len, mm_counts, mm_options)

        # Pad the dummy encoder prompt so that the profiled encoder item has
        # the maximum encoder length after the image tokens are expanded.
        # Each image placeholder token is replaced by `image_seq_len` tokens.
        num_images = mm_counts.get("image", 0)
        expanded_len = len(inputs.prompt) + num_images * (image_seq_len - 1)

        pad_token_id = self.info.get_hf_config().pad_token_id or 0
        inputs.prompt = inputs.prompt + [pad_token_id] * max(0, seq_len - expanded_len)
        return inputs

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalEncDecInput:
        # Step 1: expand image placeholders and preprocess images in the same
        # way as Gemma 3. We call the base implementation directly because
        # `EncDecMultiModalProcessor.apply` would wrap the result again.
        encoder_inputs = BaseMultiModalProcessor.apply(self, inputs, timing_ctx)

        # Step 2: merge the token IDs and all images into one encoder item.
        encoder_inputs = self._pack_encoder_input(encoder_inputs, inputs)

        return self._get_enc_dec_inputs(
            prompt=inputs.prompt,
            mm_items=inputs.mm_data_items,
            encoder_inputs=encoder_inputs,
        )

    def _pack_encoder_input(
        self,
        encoder_inputs: MultiModalInput,
        processor_inputs: ProcessorInputs,
    ) -> MultiModalInput:
        prompt_token_ids = encoder_inputs["prompt_token_ids"]
        image_items = encoder_inputs["mm_kwargs"].get("image", [])

        # Each Gemma 3 image item holds one or more crops of pixel values.
        # Concatenating them keeps the order of the image soft tokens.
        # Text-only prompts get an empty tensor, so that every encoder item
        # has the same fields and items of all requests batch together.
        pixel_values = []
        for image_item in image_items:
            # The processor cache, which may drop item data, is disabled for
            # encoder-decoder models.
            assert image_item is not None
            pixel_values.append(image_item[_PIXEL_VALUES].data)
        if pixel_values:
            all_pixel_values = torch.cat(pixel_values)
        else:
            image_size = self.info.get_hf_config().encoder.vision_config.image_size
            all_pixel_values = torch.empty((0, 3, image_size, image_size))

        # The fields describe how items of different requests are batched:
        # token IDs are stacked (or listed, if lengths differ) per item, while
        # pixel values are concatenated along the crop dimension.
        item = MultiModalKwargsItem(
            {
                _ENCODER_INPUT_IDS: MultiModalFieldElem(
                    data=torch.tensor(prompt_token_ids, dtype=torch.long),
                    field=MultiModalBatchedField(),
                ),
                _PIXEL_VALUES: MultiModalFieldElem(
                    data=all_pixel_values,
                    field=MultiModalFlatField(
                        slices=[slice(0, all_pixel_values.shape[0])]
                    ),
                ),
            }
        )

        # The identity of the encoder item depends on its tokens and on the
        # identity of every image inside it.
        mm_hash = MultiModalHasher.hash_kwargs(
            self.info.ctx.get_mm_config().mm_hasher_algorithm,
            model_id=self.info.model_id,
            encoder_prompt_token_ids=np.asarray(prompt_token_ids, dtype=np.int64),
            image_hashes=list(encoder_inputs["mm_hashes"].get("image", [])),
            mm_processor_kwargs=dict(processor_inputs.hf_processor_mm_kwargs),
        )

        packed: MultiModalInput = {
            **encoder_inputs,
            "mm_kwargs": MultiModalKwargsItems({_ENCODER_MODALITY: [item]}),
            "mm_hashes": {_ENCODER_MODALITY: [mm_hash]},
            "mm_placeholders": {
                _ENCODER_MODALITY: [
                    PlaceholderRange(offset=0, length=len(prompt_token_ids))
                ]
            },
        }
        return packed


# --------------------------------------------------------------------------- #
# Attention helpers
# --------------------------------------------------------------------------- #


@dataclass
class _EncoderAttnMetadata:
    """Variable-length batch description for one encoder forward pass."""

    seq_lens: list[int]
    cu_seqlens: torch.Tensor
    max_seqlen: int


def _get_fa_version(head_dim: int) -> int | None:
    """FlashAttention version for the kernels that T5Gemma2 calls directly.

    Returns `None` when FlashAttention is unavailable (off CUDA, or on GPUs
    older than compute capability 8.0); the PyTorch path is used then.
    """
    if not current_platform.is_cuda():
        return None

    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    fa_version = get_flash_attn_version(head_size=head_dim)
    # These calls have no FA4-specific handling (e.g. its head_dim=256
    # kernel on Blackwell), and FA2 runs on the same KV cache layout.
    return 2 if fa_version == 4 else fa_version


def _encoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: _EncoderAttnMetadata,
    scale: float,
    window: tuple[int, int],
    fa_version: int | None,
) -> torch.Tensor:
    """Bidirectional attention over a variable-length batch of sequences.

    Args:
        query: [num_tokens, num_heads, head_dim]
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        attn_metadata: Lengths of the sequences in the batch.
        scale: Softmax scale.
        window: `(left, right)` local window, `(-1, -1)` for global attention.
        fa_version: FlashAttention version, or `None` to use PyTorch SDPA.

    Returns:
        [num_tokens, num_heads, head_dim]

    """
    if fa_version is not None:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        return flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=attn_metadata.cu_seqlens,
            cu_seqlens_k=attn_metadata.cu_seqlens,
            max_seqlen_q=attn_metadata.max_seqlen,
            max_seqlen_k=attn_metadata.max_seqlen,
            softmax_scale=scale,
            causal=False,
            window_size=list(window),
            fa_version=fa_version,
        )

    outputs = []
    for q, k, v in zip(
        query.split(attn_metadata.seq_lens),
        key.split(attn_metadata.seq_lens),
        value.split(attn_metadata.seq_lens),
    ):
        mask = None
        if window != (-1, -1):
            pos = torch.arange(q.shape[0], device=q.device)
            dist = pos[:, None] - pos[None, :]  # query position - key position
            mask = (dist <= window[0]) & (-dist <= window[1])
        # SDPA expects [batch, heads, seq, head_dim].
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            scale=scale,
            enable_gqa=True,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


def _get_layer_context(
    layer_name: str,
) -> tuple[Any, Attention, torch.Tensor, torch.Tensor | None]:
    """Return `(attn_metadata, layer, kv_cache, slot_mapping)` of a layer."""
    forward_context = get_forward_context()
    all_metadata: Any = forward_context.attn_metadata
    if isinstance(all_metadata, list):
        all_metadata = all_metadata[0]
    attn_metadata = (
        all_metadata.get(layer_name) if isinstance(all_metadata, dict) else None
    )
    layer = forward_context.no_compile_layers[layer_name]
    all_slots: Any = forward_context.slot_mapping
    slot_mapping = all_slots.get(layer_name) if isinstance(all_slots, dict) else None
    return attn_metadata, layer, layer.kv_cache, slot_mapping


def _flash_attn_with_lse(
    layer: Attention,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    fa_version: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paged attention on a FlashAttention-backend cache, returning LSE."""
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    impl = cast(FlashAttentionImpl, layer.impl)
    # The metadata builder schedules ahead of time for the backend's own FA
    # version only.
    scheduler_metadata = (
        attn_metadata.scheduler_metadata
        if fa_version == impl.vllm_flash_attn_version
        else None
    )
    causal = attn_metadata.causal
    assert isinstance(causal, bool), "T5Gemma2 does not support dynamic causal"

    # Logical cache layout: [num_blocks, num_kv_heads, block_size, 2*head_dim].
    key_cache, value_cache = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    value_cache = canonicalize_singleton_dim_strides(value_cache)

    # Only the self-attention part is causal; the cross part sees the full
    # encoder output and has no sliding window.
    window = list(impl.sliding_window) if causal else [-1, -1]
    out, lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=torch.empty_like(query),
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=attn_metadata.seq_lens,
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=causal,
        window_size=window,
        block_table=attn_metadata.block_table,
        softcap=impl.logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=scheduler_metadata,
        fa_version=fa_version,
        num_splits=attn_metadata.max_num_splits,
    )
    return out, lse


def _write_kv_cache_torch(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store K/V using the logical cache layout `[B, H, N, 2*D]`.

    Only `_paged_attention_with_lse_torch` reads caches written here, so the
    backend-specific packing of the kernels is irrelevant.
    """
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[2]
    valid = slot_mapping >= 0  # Padded tokens have slot -1.
    slots = slot_mapping[valid]
    kv = torch.cat([key[:num_tokens], value[:num_tokens]], dim=-1)[valid]
    # Two index tensors separated by a slice: the result is [num_slots, H, 2D].
    kv_cache[slots // block_size, :, slots % block_size] = kv.to(kv_cache.dtype)


def _paged_attention_with_lse_torch(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    scale: float,
    sliding_window: int | None,
    softcap: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference paged attention for platforms without FlashAttention."""
    num_tokens, num_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = kv_cache.shape
    causal = bool(attn_metadata.causal)

    out = torch.zeros_like(query)
    lse = torch.full(
        (num_heads, num_tokens), float("-inf"), dtype=torch.float32, device=query.device
    )
    query_start_loc = attn_metadata.query_start_loc.tolist()
    seq_lens = attn_metadata.seq_lens.tolist()
    for i, kv_len in enumerate(seq_lens):
        q_start, q_end = query_start_loc[i], query_start_loc[i + 1]
        if q_start == q_end or kv_len == 0:
            continue
        blocks = attn_metadata.block_table[i, : cdiv(kv_len, block_size)]
        # [num_blocks, H, N, 2D] -> [num_blocks * N, H, 2D] -> keep kv_len rows
        kv = kv_cache[blocks].transpose(1, 2).flatten(0, 1)[:kv_len].float()
        k = kv[..., :head_dim].repeat_interleave(num_heads // num_kv_heads, dim=1)
        v = kv[..., head_dim:].repeat_interleave(num_heads // num_kv_heads, dim=1)

        q = query[q_start:q_end].float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        if softcap:
            scores = softcap * torch.tanh(scores / softcap)
        if causal:
            # The queries are the last `q_len` tokens of the sequence.
            q_pos = torch.arange(kv_len - (q_end - q_start), kv_len, device=q.device)
            k_pos = torch.arange(kv_len, device=q.device)
            mask = k_pos[None, :] <= q_pos[:, None]
            if sliding_window is not None:
                mask &= k_pos[None, :] > q_pos[:, None] - sliding_window
            scores = scores.masked_fill(~mask, float("-inf"))

        lse_i = torch.logsumexp(scores, dim=-1)  # [H, q_len]
        probs = torch.exp(scores - lse_i[..., None])
        out[q_start:q_end] = torch.einsum("hqk,khd->qhd", probs, v).to(out.dtype)
        lse[:, q_start:q_end] = lse_i
    return out, lse


def t5gemma2_merged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cross_key: torch.Tensor | None,
    cross_value: torch.Tensor | None,
    output: torch.Tensor,
    self_layer_name: str,
    cross_layer_name: str,
    fa_version: int | None,
    scale: float,
    softcap: float | None,
) -> None:
    """One softmax over the decoder self-attention keys and the encoder keys.

    Args:
        query: [num_tokens, num_heads, head_dim]
        key: Decoder keys of the scheduled tokens,
            [num_tokens, num_kv_heads, head_dim].
        value: Decoder values of the scheduled tokens.
        cross_key: Encoder keys of the requests in their first decoder step,
            or `None` when every request is decoding.
        cross_value: Encoder values, like `cross_key`.
        output: [num_tokens, num_heads, head_dim], written in place.
        self_layer_name: Layer that owns the decoder self-attention cache.
        cross_layer_name: Layer that owns the cross-attention cache.
        fa_version: FlashAttention version to use on FlashAttention-backend
            caches, or `None` for the PyTorch reference implementation.
        scale: Softmax scale, used by the PyTorch implementation.
        softcap: Attention logit soft cap, used by the PyTorch implementation.

    """
    self_metadata, self_layer, self_kv_cache, self_slots = _get_layer_context(
        self_layer_name
    )
    cross_metadata, cross_layer, cross_kv_cache, _ = _get_layer_context(
        cross_layer_name
    )
    if self_metadata is None or cross_metadata is None:
        # Profiling run without attention metadata.
        output.zero_()
        return

    num_tokens = self_metadata.num_actual_tokens
    query = query[:num_tokens]

    # 1. Store the new K/V. The cross-attention metadata builder computes the
    #    slots of the encoder tokens and stores them on the metadata.
    if fa_version is not None:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        if self_slots is not None:
            cast(FlashAttentionImpl, self_layer.impl).do_kv_cache_update(
                self_layer, key, value, self_kv_cache, self_slots
            )
        if cross_key is not None:
            cast(FlashAttentionImpl, cross_layer.impl).do_kv_cache_update(
                cross_layer,
                cross_key,
                cross_value,
                cross_kv_cache,
                cross_metadata.slot_mapping,
            )
    else:
        if self_slots is not None:
            _write_kv_cache_torch(self_kv_cache, key, value, self_slots)
        if cross_key is not None:
            _write_kv_cache_torch(
                cross_kv_cache, cross_key, cross_value, cross_metadata.slot_mapping
            )

    # 2. Attend to both caches separately, keeping the softmax denominators.
    if fa_version is not None:
        self_out, self_lse = _flash_attn_with_lse(
            self_layer, query, self_kv_cache, self_metadata, fa_version
        )
        cross_out, cross_lse = _flash_attn_with_lse(
            cross_layer, query, cross_kv_cache, cross_metadata, fa_version
        )
    else:
        self_out, self_lse = _paged_attention_with_lse_torch(
            query,
            self_kv_cache,
            self_metadata,
            scale,
            self_layer.sliding_window,
            softcap,
        )
        cross_out, cross_lse = _paged_attention_with_lse_torch(
            query, cross_kv_cache, cross_metadata, scale, None, softcap
        )

    # 3. softmax([s1, s2]) @ [v1; v2] == LSE-weighted mix of both outputs.
    merge_attn_states(output[:num_tokens], self_out, self_lse, cross_out, cross_lse)


@contextmanager
def _lora_mapping(
    layer: nn.Module, punica_wrapper: "PunicaWrapperBase | None"
) -> Iterator[None]:
    """Run a LoRA layer with another token-to-adapter mapping."""
    if punica_wrapper is None or not isinstance(layer, BaseLayerWithLoRA):
        yield
        return
    original = layer.punica_wrapper
    layer.set_mapping(punica_wrapper)
    try:
        yield
    finally:
        layer.set_mapping(original)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class T5Gemma2TextEmbedding(VocabParallelEmbedding):
    """Gemma embedding scaled by sqrt(hidden_size), plus a learned EOI vector.

    HF replaces the embedding of the end-of-image token by `eoi_embedding`
    (without scaling) in both the encoder and the decoder.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        eoi_token_index: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_embeddings, embedding_dim, quant_config=quant_config, prefix=prefix
        )
        self.eoi_token_index = eoi_token_index
        self.eoi_embedding = nn.Parameter(torch.zeros(embedding_dim))
        # Created under the model dtype, like `embed_scale.to(weight.dtype)`
        # in HF, so the scaling rounds the same way.
        self.register_buffer(
            "normalizer", torch.tensor(embedding_dim**0.5), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = super().forward(input_ids) * self.normalizer
        is_eoi = (input_ids == self.eoi_token_index).unsqueeze(-1)
        return torch.where(is_eoi, self.eoi_embedding.to(embeds.dtype), embeds)


class T5Gemma2AttentionBase(nn.Module):
    """Projections, q/k norms and RoPE shared by encoder and decoder layers."""

    def __init__(
        self,
        config: T5Gemma2TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_type = config.layer_types[extract_layer_index(prefix)]
        self.is_sliding = layer_type == "sliding_attention"
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters[layer_type],
            is_neox_style=True,
        )

    def project_qkv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q/k (normed, with RoPE) and v, all of shape [T, heads, D]."""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        q, k = self.rotary_emb(positions, q.flatten(-2), k.flatten(-2))
        return (
            q.unflatten(-1, (self.num_heads, self.head_dim)),
            k.unflatten(-1, (self.num_kv_heads, self.head_dim)),
            v.unflatten(-1, (self.num_kv_heads, self.head_dim)),
        )


class T5Gemma2EncoderAttention(T5Gemma2AttentionBase):
    def __init__(
        self,
        config: T5Gemma2TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        if self.is_sliding:
            # HF `sliding_window_mask_function(is_causal=False)` keeps keys
            # with -sw//2 <= (q - k) < (sw + 1)//2, i.e. this (left, right).
            sliding_window = config.sliding_window
            self.window = ((sliding_window + 1) // 2 - 1, sliding_window // 2)
        else:
            self.window = (-1, -1)
        self.fa_version = _get_fa_version(self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: _EncoderAttnMetadata,
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(positions, hidden_states)
        attn_output = _encoder_attention(
            q, k, v, attn_metadata, self.scaling, self.window, self.fa_version
        )
        output, _ = self.o_proj(attn_output.flatten(-2))
        return output


class T5Gemma2MergedAttention(T5Gemma2AttentionBase):
    """Decoder attention over both its own tokens and the encoder outputs."""

    def __init__(
        self,
        config: T5Gemma2TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.softcap = config.attn_logit_softcapping

        # With FlashAttention, both caches must use its backend, because
        # `_flash_attn_with_lse` calls the kernels on them directly. Without
        # it, the PyTorch path reads and writes the caches itself, so any
        # backend works; on CUDA we pick Triton, which supports every type
        # of attention layer.
        self.fa_version = _get_fa_version(self.head_dim)
        attn_backend: type[AttentionBackend] | None = None
        if self.fa_version is not None:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

            attn_backend = FlashAttentionBackend
        elif current_platform.is_cuda():
            from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

            attn_backend = TritonAttentionBackend

        # The two layers below only own a KV cache each. Their `forward` is
        # never called; `t5gemma2_merged_attention` reads both caches.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=config.attn_logit_softcapping,
            per_layer_sliding_window=(
                config.sliding_window if self.is_sliding else None
            ),
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
            attn_backend=attn_backend,
        )
        self.cross_attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=config.attn_logit_softcapping,
            prefix=f"{prefix}.cross_attn",
            attn_type=AttentionType.ENCODER_DECODER,
            underlying_attn_backend=attn_backend,
        )
        for layer in (self.attn, self.cross_attn):
            if layer.kv_cache_dtype != "auto":
                raise NotImplementedError(
                    "T5Gemma2 does not support quantized KV cache yet."
                )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        encoder_lora_mapping: "PunicaWrapperBase | None",
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(positions, hidden_states)

        cross_k = cross_v = None
        if encoder_hidden_states is not None:
            # The encoder K/V reuse k_proj/v_proj and k_norm, without RoPE.
            with _lora_mapping(self.qkv_proj, encoder_lora_mapping):
                qkv, _ = self.qkv_proj(encoder_hidden_states)
            _, cross_k, cross_v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            cross_k = self.k_norm(cross_k.unflatten(-1, (self.num_kv_heads, -1)))
            cross_v = cross_v.unflatten(-1, (self.num_kv_heads, self.head_dim))

        attn_output = torch.empty_like(q)
        t5gemma2_merged_attention(
            q,
            k,
            v,
            cross_k,
            cross_v,
            attn_output,
            self.attn.layer_name,
            self.cross_attn.layer_name,
            self.fa_version,
            self.scaling,
            self.softcap,
        )
        output, _ = self.o_proj(attn_output.flatten(-2))
        return output


class T5Gemma2EncoderLayer(nn.Module):
    def __init__(
        self,
        config: T5Gemma2TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = T5Gemma2EncoderAttention(
            config, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        eps = config.rms_norm_eps
        self.pre_self_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_self_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: _EncoderAttnMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, attn_metadata)
        hidden_states = residual + self.post_self_attn_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.post_feedforward_layernorm(hidden_states)


class T5Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: T5Gemma2TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = T5Gemma2MergedAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        eps = config.rms_norm_eps
        self.pre_self_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_self_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
        encoder_lora_mapping: "PunicaWrapperBase | None",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `norm(x, residual)` fuses the residual add: it returns
        # (norm(x + residual), x + residual).
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_self_attn_layernorm(hidden_states)
        else:
            hidden_states, residual = self.pre_self_attn_layernorm(
                hidden_states, residual
            )
        hidden_states = self.self_attn(
            positions, hidden_states, encoder_hidden_states, encoder_lora_mapping
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


class T5Gemma2TextEncoder(nn.Module):
    def __init__(
        self,
        config: T5Gemma2TextConfig,
        eoi_token_index: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = T5Gemma2TextEmbedding(
            config.vocab_size,
            config.hidden_size,
            eoi_token_index,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            T5Gemma2EncoderLayer(config, quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: _EncoderAttnMetadata,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, attn_metadata)
        return self.norm(hidden_states)


class T5Gemma2Encoder(nn.Module):
    """Text encoder plus the SigLIP vision tower that feeds it."""

    def __init__(
        self,
        config: T5Gemma2EncoderConfig,
        eoi_token_index: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.image_token_id = config.image_token_index
        self.text_model = T5Gemma2TextEncoder(
            config.text_config,
            eoi_token_index,
            quant_config=quant_config,
            prefix=f"{prefix}.text_model",
        )
        self.vision_tower = SiglipVisionModel(
            config.vision_config,
            quant_config,
            prefix=f"{prefix}.vision_tower",
        )
        # The Gemma 3 projector only reads `vision_config`, `text_config` and
        # `mm_tokens_per_image`, which the encoder config also has.
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

    def forward(
        self,
        encoder_input_ids: list[torch.Tensor],
        pixel_values: torch.Tensor,
    ) -> list[torch.Tensor]:
        seq_lens = [ids.numel() for ids in encoder_input_ids]
        input_ids = torch.cat(encoder_input_ids)
        # Every sequence restarts at position 0.
        positions = torch.cat(
            [torch.arange(n, device=input_ids.device) for n in seq_lens]
        )
        cu_seqlens = torch.tensor(
            [0, *np.cumsum(seq_lens)], dtype=torch.int32, device=input_ids.device
        )
        attn_metadata = _EncoderAttnMetadata(seq_lens, cu_seqlens, max(seq_lens))

        inputs_embeds = self.text_model.embed_tokens(input_ids)
        if pixel_values.shape[0] > 0:
            dtype = self.vision_tower.dtype
            image_features = self.vision_tower(pixel_values.to(dtype))
            image_embeds = self.multi_modal_projector(image_features)
            is_image = (input_ids == self.image_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                is_image, image_embeds.to(inputs_embeds.dtype)
            )

        hidden_states = self.text_model(inputs_embeds, positions, attn_metadata)
        return list(hidden_states.split(seq_lens))


class T5Gemma2Decoder(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        embed_tokens: T5Gemma2TextEmbedding | None = None,
    ) -> None:
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        decoder_config = config.decoder
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if embed_tokens is None:
            embed_tokens = T5Gemma2TextEmbedding(
                decoder_config.vocab_size,
                decoder_config.hidden_size,
                config.eoi_token_index,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(
            T5Gemma2DecoderLayer(
                decoder_config,
                cache_config,
                quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(decoder_config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(
            decoder_config.hidden_size, eps=decoder_config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_lora_mapping: "PunicaWrapperBase | None" = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                encoder_hidden_states,
                encoder_lora_mapping,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class T5Gemma2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        self.encoder = T5Gemma2Encoder(
            config.encoder,
            config.eoi_token_index,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.encoder",
        )
        # The encoder and decoder share one embedding module, including the
        # EOI vector (`_tied_weights_keys` of `T5Gemma2Model` in HF).
        self.decoder = T5Gemma2Decoder(
            vllm_config=vllm_config,
            prefix=f"{prefix}.decoder",
            embed_tokens=self.encoder.text_model.embed_tokens,
        )


@MULTIMODAL_REGISTRY.register_processor(
    T5Gemma2MultiModalProcessor,
    info=T5Gemma2ProcessingInfo,
    dummy_inputs=T5Gemma2DummyInputsBuilder,
)
class T5Gemma2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsLoRA):
    # Required for LoRA on the text encoder (see the module docstring), so
    # `T5Gemma2ForConditionalGenerationConfig` enables it with LoRA.
    supports_tower_connector_lora = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Checkpoints store the text encoder without the `text_model.` level that
    # HF adds at load time (see `t5gemma2_encoder` in HF's
    # `conversion_mapping.py`); accept both spellings.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            re.compile(r"^model\.encoder\.(embed_tokens|layers|norm)\."): (
                r"model.encoder.text_model.\1."
            ),
        },
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
        orig_to_new_prefix={
            # The LM head is tied to the shared embedding.
            "lm_head.out_proj.": "lm_head.",
        },
    )

    # vLLM's `SiglipVisionModel` keeps the `vision_model.` level, which newer
    # HF versions drop when saving.
    vision_hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            re.compile(r"^model\.encoder\.vision_tower\.(?!vision_model\.)"): (
                "model.encoder.vision_tower.vision_model."
            ),
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<start_of_image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: T5Gemma2Config = vllm_config.model_config.hf_config
        self.config = config

        mm_config = vllm_config.model_config.get_multimodal_config()
        if mm_config.get_limit_per_prompt(_ENCODER_MODALITY) == 0:
            raise ValueError(
                "T5Gemma2 runs its encoder through the multimodal encoder path, "
                f"so neither `language_model_only` nor a zero limit for the "
                f"{_ENCODER_MODALITY!r} modality is supported. Use "
                "`limit_mm_per_prompt={'image': 0}` for text-only inputs."
            )
        if (
            current_platform.is_cuda()
            and _get_fa_version(config.decoder.head_dim) is None
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            raise ValueError(
                "T5Gemma2 needs FlashAttention for CUDA graphs on this GPU; "
                "pass `enforce_eager=True`."
            )

        with self._mark_composite_model(
            vllm_config,
            language_targets=T5Gemma2Decoder,
            tower_targets={"image": (SiglipVisionModel, Gemma3MultiModalProjector)},
        ):
            self.model = T5Gemma2Model(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )

        decoder_config = config.decoder
        self.lm_head = ParallelLMHead(
            decoder_config.vocab_size,
            decoder_config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.lm_head = self.lm_head.tie_weights(self.model.decoder.embed_tokens)
        self.logits_processor = LogitsProcessor(
            decoder_config.vocab_size,
            soft_cap=decoder_config.final_logit_softcapping,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Only used for the decoder; the encoder embeds its own tokens.
        return self.model.decoder.embed_input_ids(input_ids)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Run the encoder for a batch of packed encoder items."""
        encoder_input_ids = kwargs[_ENCODER_INPUT_IDS]
        if isinstance(encoder_input_ids, torch.Tensor):
            # Items of equal length are stacked into [num_items, seq_len].
            encoder_input_ids = list(encoder_input_ids.unbind(0))
        pixel_values = kwargs[_PIXEL_VALUES]
        assert isinstance(pixel_values, torch.Tensor)
        return self.model.encoder(encoder_input_ids, pixel_values)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model=_LORA_DECODER_PREFIX,
            tower_model=_LORA_ENCODER_PREFIX,
        )

    def get_mm_lora_token_counts(
        self,
        *,
        modality: str,
        mm_kwargs: MultiModalKwargsItem | None,
        num_mm_embeds: int,
    ) -> tuple[int, int | None]:
        # The text encoder runs on every token of the packed encoder item.
        return num_mm_embeds, None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Encoder outputs are only passed in the first decoder step of each
        # request; afterwards their K/V live in the cross-attention cache.
        encoder_hidden_states = None
        encoder_lora_mapping = None
        if encoder_outputs:
            encoder_hidden_states = torch.cat(encoder_outputs)
            # The model runner maps these tokens, in this order, to their
            # adapters when it runs the encoder in the same step.
            lora_manager = getattr(self, "lora_manager", None)
            if lora_manager is not None:
                encoder_lora_mapping = lora_manager.punica_wrapper_mapping[
                    _LORA_ENCODER_PREFIX
                ]
        return self.model.decoder(
            input_ids, positions, encoder_hidden_states, encoder_lora_mapping
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The SigLIP tower stacks its own q/k/v in `SiglipVisionModel`, so the
        # text stacking below must not rename its weights. The vision weights
        # are small, so they are buffered while the rest is streamed.
        vision_weights: list[tuple[str, torch.Tensor]] = []

        def text_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith("model.encoder.vision_tower."):
                    vision_weights.append((name, weight))
                else:
                    yield name, weight

        loaded = AutoWeightsLoader(self).load_weights(
            text_weights(), mapper=self.hf_to_vllm_mapper
        )
        loaded |= AutoWeightsLoader(self).load_weights(
            vision_weights, mapper=self.vision_hf_to_vllm_mapper
        )
        return loaded
