# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForSeq2SeqLM

from tests.models.utils import check_logprobs_close
from vllm.assets.image import ImageAsset
from vllm.model_executor.models.t5gemma2 import (
    _paged_attention_with_lse_torch,
    _write_kv_cache_torch,
)
from vllm.multimodal.image import convert_image_mode
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from ....conftest import HfRunner, VllmRunner

MODEL = "google/t5gemma-2-270m-270m"

TEXT_PROMPTS = [
    "Summarize: The quick brown fox jumps over the lazy dog near the river bank.",
    "Translate English to French: Where is the train station?",
]
IMAGE_PROMPT = "<start_of_image> in this image, there is"


def _make_paged_kv(
    keys: list[torch.Tensor], values: list[torch.Tensor], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Store per-sequence K/V in a shuffled paged cache; return cache, table."""
    num_kv_heads, head_dim = keys[0].shape[1:]
    max_blocks = max((k.shape[0] + block_size - 1) // block_size for k in keys)
    num_blocks = len(keys) * max_blocks
    kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * head_dim)
    block_ids = torch.randperm(num_blocks).view(len(keys), max_blocks)
    for i, (k, v) in enumerate(zip(keys, values)):
        pos = torch.arange(k.shape[0])
        slots = block_ids[i, pos // block_size] * block_size + pos % block_size
        _write_kv_cache_torch(kv_cache, k, v, slots)
    return kv_cache, block_ids


@pytest.mark.core_model
@pytest.mark.cpu_model
@pytest.mark.parametrize("sliding_window", [None, 3])
def test_merged_attention_is_one_softmax(sliding_window: int | None) -> None:
    """Merging the self and cross partial attentions by their LSE must equal a
    single softmax over the concatenated keys, as in HF `T5Gemma2MergedAttention`.
    """
    torch.manual_seed(0)
    num_heads, num_kv_heads, head_dim, block_size = 4, 2, 8, 4
    scale = head_dim**-0.5
    # Two requests: one in its first step with 3 decoder tokens, and one
    # decoding its 6th token. Encoder lengths are 5 and 7.
    dec_lens, query_lens, enc_lens = [3, 6], [3, 1], [5, 7]

    self_k = [torch.randn(n, num_kv_heads, head_dim) for n in dec_lens]
    self_v = [torch.randn(n, num_kv_heads, head_dim) for n in dec_lens]
    cross_k = [torch.randn(n, num_kv_heads, head_dim) for n in enc_lens]
    cross_v = [torch.randn(n, num_kv_heads, head_dim) for n in enc_lens]
    query = torch.randn(sum(query_lens), num_heads, head_dim)
    query_start_loc = torch.tensor([0, *torch.tensor(query_lens).cumsum(0)])

    def partial(keys, values, seq_lens, causal, window):
        kv_cache, block_table = _make_paged_kv(keys, values, block_size)
        metadata = SimpleNamespace(
            query_start_loc=query_start_loc,
            seq_lens=torch.tensor(seq_lens),
            block_table=block_table,
            causal=causal,
        )
        return _paged_attention_with_lse_torch(
            query, kv_cache, metadata, scale, window, softcap=None
        )

    self_out, self_lse = partial(self_k, self_v, dec_lens, True, sliding_window)
    cross_out, cross_lse = partial(cross_k, cross_v, enc_lens, False, None)
    merged = torch.empty_like(query)
    merge_attn_states(merged, self_out, self_lse, cross_out, cross_lse)

    group = num_heads // num_kv_heads
    for i, (q_len, dec_len) in enumerate(zip(query_lens, dec_lens)):
        q = query[query_start_loc[i] : query_start_loc[i + 1]]
        k = torch.cat([self_k[i], cross_k[i]]).repeat_interleave(group, dim=1)
        v = torch.cat([self_v[i], cross_v[i]]).repeat_interleave(group, dim=1)
        q_pos = torch.arange(dec_len - q_len, dec_len)[:, None]
        k_pos = torch.arange(dec_len)[None, :]
        self_mask = k_pos <= q_pos
        if sliding_window is not None:
            self_mask &= k_pos > q_pos - sliding_window
        mask = torch.cat([self_mask, torch.ones(q_len, enc_lens[i]).bool()], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        expected = torch.einsum("hqk,khd->qhd", probs, v)
        torch.testing.assert_close(
            merged[query_start_loc[i] : query_start_loc[i + 1]], expected
        )


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    prompts: list[str],
    images: list | None,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Greedy generation must match HF for text-only and image prompts."""
    with vllm_runner(
        MODEL,
        dtype=dtype,
        max_model_len=2048,
        # Text-only runs disable images, which also skips the vision tower.
        limit_mm_per_prompt={"image": 0 if images is None else len(images)},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs=num_logprobs, images=images
        )

    with hf_runner(MODEL, dtype=dtype, auto_cls=AutoModelForSeq2SeqLM) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            prompts, max_tokens, num_logprobs=num_logprobs, images=images
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("num_logprobs", [5])
def test_text_only(hf_runner, vllm_runner, dtype: str, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        TEXT_PROMPTS,
        None,
        dtype=dtype,
        max_tokens=32,
        num_logprobs=num_logprobs,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("num_logprobs", [5])
def test_image(hf_runner, vllm_runner, dtype: str, num_logprobs: int) -> None:
    images = [
        convert_image_mode(ImageAsset(name).pil_image, "RGB")
        for name in ("stop_sign", "cherry_blossom")
    ]
    run_test(
        hf_runner,
        vllm_runner,
        [IMAGE_PROMPT] * len(images),
        images,
        dtype=dtype,
        max_tokens=32,
        num_logprobs=num_logprobs,
    )
