from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from layers import RMSNorm, Embedding
from attention import attention_monstrosity, RoPEEmbedding, Mask, make_rope_mask_indices


class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply the OpenELMRMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the OpenELMRMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying OpenELMRMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"
        )


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    return (x * pos_cos) + (_rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.
    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.
    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(
        self, model_dim: int, max_seq_length: int, freq_constant: int = 10000
    ) -> None:
        inv_freq = 1.0 / (
            freq_constant
            ** (torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim)
        )
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_device: torch.device = torch.device("cpu"),
        key_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute sine and cos embeddings.
        Args:
            key_len: Number of tokens in the key embeddings in the transformer model.
            device: Device where the key embeddings are stored.
            key_dtype: Data type of the key embeddings.
        Returns:
            None
        ...note:
            We recalculate the sine and cosine embeddings if any of the following conditions are met:
                1. The number of tokens in key embeddings are greater than the cached sequence length.
                2. Sine and cosine caches are empty.
                3. The device and data type of sine and cosine embeddings does not match with the key embeddings.
        """
        if (
            key_len > self._cached_seq_length
            or self._cached_cos is None
            or (self._cached_cos is not None and self._cached_cos.device != key_device)
            or (self._cached_cos is not None and self._cached_cos.dtype != key_dtype)
            or self._cached_sin is None
            or (self._cached_sin is not None and self._cached_sin.device != key_device)
            or (self._cached_sin is not None and self._cached_sin.dtype != key_dtype)
        ):
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            # The shape of 'pos_index' is [number of key tokens]
            pos_index = torch.arange(
                self._cached_seq_length,
                dtype=torch.float32,
                device=self.inv_freq.device,
            )
            # The shape of 'pos_index_theta' is [number of key tokens, model dimension]
            pos_index_theta = torch.einsum("i,j->ij", pos_index, self.inv_freq)
            # The shape of 'emb' is [number of key tokens, model dimension]
            emb = torch.cat((pos_index_theta, pos_index_theta), dim=-1)

            # the shape of cos and sin embeddings is [number of key tokens, model_dim]
            cos_emb = emb.cos().to(dtype=key_dtype, device=key_device)
            sin_emb = emb.sin().to(dtype=key_dtype, device=key_device)

            # the shape of cached cos and sin embeddings is [1, 1, number of key tokens, model_dim]
            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function of RoPE embeddings.
        Args:
            query: Query embeddings in the transformer model. The shape of query embeddings is
                [Batch, number of query heads, number of query tokens, model dimension].
            key: Key embeddings in the transformer model. The shape of key embeddings is
                [Batch, number of key heads, number of key tokens, model dimension].
        Returns:
            A tuple containing the query and key embeddings with positional information. The shape of the returned query
            and key embeddings is the same as the input query and key embeddings respectively.
        ...note:
            The RoPE embedding computation is done in full-precision. After the computation, input query and key tensors
            are casted to original input datatype.
        """
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.device == query.device
        assert key.dtype == query.dtype

        # In the context of self-attention, the lengths of keys and queries are equal.
        # However, in generation tasks, such as predicting the next token in a sequence, the lengths of keys and queries
        # can differ. For instance, when employing key-value (KV) caching for sequence prediction, the keys
        # represent embeddings of previous tokens and the current token, while the query corresponds
        # to the embedding of the current token only.
        assert (
            key_len >= query_len
        ), "Number of keys has to be greater than or equal to number of queries."

        query_float = query.float()
        key_float = key.float()

        self._compute_sin_cos_embeddings(
            key_len, key_device=key_float.device, key_dtype=key_float.dtype
        )
        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        return query_float.type_as(query), key_float.type_as(key)


class OpenELMMultiHeadCausalAttention(nn.Module):
    def __init__(
        self,
        head_dim,
        # model_dim,
        num_query_heads,
        num_kv_heads,
        normalize_qk_projections,
        use_pos_embedding=True,
        rope_max_length=2048,
        rope_freq_constant=10_000,
    ) -> None:
        super().__init__()
        head_dim = head_dim
        q_heads = num_query_heads
        k_heads = num_kv_heads
        v_heads = num_kv_heads

        if use_pos_embedding:
            self.pos_embedding = OpenELMRotaryEmbedding(
                model_dim=head_dim,
                max_seq_length=rope_max_length,
                freq_constant=rope_freq_constant,
            )
        else:
            self.pos_embedding = None

        if normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # self.qkv_proj = nn.Linear(
        #     in_features=config.model_dim,
        #     out_features=(q_heads + k_heads + v_heads) * head_dim,
        #     bias=False,
        # )
        # self.out_proj = nn.Linear(
        #     in_features=q_heads * head_dim,
        #     out_features=config.model_dim,
        #     bias=False,
        # )

        self.head_dim = head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        # self.transformer_dim = model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of multi-head self-attention.
        Args:
            hidden_states: Input tensor of the shape [batch size, sequence length, model dimension].
            past_key_value: Tensor storing the cached keys and values.
            output_attentions: output attention weights.
            use_cache: Specifies whether to use kv-cache for generation.
            cache_position: used for updating the kv-cache.
        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """

        # scaled_dot_product_attention does not return attention weights, set output_attentions to False
        output_attentions = False
        batch_size, seq_length, d_model = hidden_states.size()

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        # qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = hidden_states.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose(1, 2)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        # past_key_value = getattr(self, "past_key_value", past_key_value)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; position_ids needed for the static cache
        #     # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     cache_kwargs = {"cache_position": cache_position}
        #     keys, values = past_key_value.update(
        #         keys, values, self.layer_idx, cache_kwargs
        #     )

        # Add positional embedding
        if self.pos_embedding:
            queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # GQA
            # [B, k_h, S, h] --> [B, q_h, S, h]
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            # [B, v_h, S, h] --> [B, q_h, S, h]
            values = values.repeat_interleave(self.num_groups, dim=1)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        #     causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=causal_mask,
            dropout_p=0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        # attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, queries, keys


def main():
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import coremltools.converters.mil as mil

    np.random.seed(42)
    torch.manual_seed(42)
    bsz, seqlen, dim = 1, 128, 640
    headdim, nqheads, nkvheads = 64, 6, 2
    # We use embedding because rope and mask with flexible shape require input ids instead of
    # hidden states as input
    torch_causal_mask = torch.full(
        (128, 128),
        fill_value=True,
        dtype=torch.bool,
    )
    torch_causal_mask = torch.triu(torch_causal_mask, diagonal=1)
    min_dtype = torch.finfo(torch.float32).min
    torch_causal_mask = (
        torch_causal_mask[None, None, :, :].repeat(bsz, 1, 1, 1).to(torch.float32)
        * min_dtype
    )
    input_ids = np.random.randint(0, 128, (bsz, seqlen), dtype=np.int32)
    torch_input_ids = torch.from_numpy(input_ids)
    embedding = nn.Embedding(seqlen, dim)
    with torch.no_grad():
        torch_hidden_states = embedding(torch_input_ids)

    # Used in CoreML to build flexible models
    seqlens = [128, 256]
    bsizes = [1]
    cml_embedding = Embedding(embedding.weight.detach().numpy(), name="embedding")
    cml_mask = Mask(256)

    for do_pre_norm in [False, True]:
        for do_rope in [False, True]:
            for do_norm in [False, True]:
                for channels_first in [False, True]:
                    for multi_query_head in [False, True]:
                        for repeat_interleave in [True, False]:

                            np.random.seed(42)
                            torch.manual_seed(42)

                            torch_attn = OpenELMMultiHeadCausalAttention(
                                headdim,
                                nqheads,
                                nkvheads,
                                normalize_qk_projections=do_norm,
                                use_pos_embedding=do_rope,
                            )
                            with torch.no_grad():
                                torch_results = torch_attn(
                                    torch_hidden_states, torch_causal_mask
                                )  # [0].numpy()
                                torch_result = torch_results[0].numpy()

                            shapes = [
                                (bsz, seqlen) for seqlen in seqlens for bsz in bsizes
                            ]
                            shape = mil.input_types.EnumeratedShapes(shapes=shapes)

                            if do_norm:
                                eps = torch_attn.q_norm.eps
                                qnormw = torch_attn.q_norm.weight.detach().numpy()
                                if channels_first:
                                    qnormw = np.expand_dims(qnormw, -1)
                                qnorm = RMSNorm(
                                    qnormw,
                                    eps,
                                    axes=[1] if channels_first else [2],
                                )
                                knormw = torch_attn.k_norm.weight.detach().numpy()
                                if channels_first:
                                    knormw = np.expand_dims(knormw, -1)
                                knorm = RMSNorm(
                                    knormw,
                                    eps,
                                    axes=[1] if channels_first else [2],
                                )
                            else:
                                qnorm = None
                                knorm = None

                            if do_rope:
                                rope = RoPEEmbedding(headdim, 2048, 2048)
                            else:
                                rope = None

                            @mb.program(
                                input_specs=[
                                    mb.TensorSpec(
                                        shape.symbolic_shape,
                                        dtype=mil.input_types.types.int32,
                                    ),
                                ],
                                opset_version=mil.builder.AvailableTarget.iOS17,
                            )
                            def cml_program(input_ids):
                                indices = make_rope_mask_indices(input_ids)
                                if do_rope:
                                    rope.init_embedding_slices(
                                        indices, channels_first=channels_first
                                    )
                                mask = cml_mask.get_mask(indices)
                                hidden_states = cml_embedding(input_ids)
                                if channels_first:
                                    hidden_states = mb.transpose(
                                        x=hidden_states, perm=[0, 2, 1]
                                    )
                                attn_result = attention_monstrosity(
                                    hidden_states,
                                    mask,
                                    headdim,
                                    nqheads,
                                    nkvheads,
                                    qnorm=qnorm,
                                    knorm=knorm,
                                    rope=rope,
                                    channels_first=channels_first,
                                    pre_normalization_and_pos_encoding=do_pre_norm,
                                    multi_query_head=multi_query_head,
                                    repeat_interleave=repeat_interleave,
                                )
                                x = mb.identity(
                                    x=attn_result[0],
                                    name="coreml_attention_output",
                                )  # Idendity for constant naming
                                return x, *attn_result[1:]

                            cml_attention_model = ct.convert(
                                cml_program,
                                compute_units=ct.ComputeUnit.CPU_ONLY,
                                compute_precision=ct.precision.FLOAT32,
                                minimum_deployment_target=ct.target.iOS17,
                                inputs=[
                                    ct.TensorType(
                                        name="input_ids",
                                        shape=ct.EnumeratedShapes(shapes),
                                    ),
                                ],
                                # pass_pipeline=ct.PassPipeline.EMPTY,
                            )

                            cml_attn_results = cml_attention_model.predict(
                                {"input_ids": input_ids}
                            )
                            cml_attn_result = cml_attn_results[
                                "coreml_attention_output"
                            ]
                            if channels_first:
                                cml_attn_result = cml_attn_result.transpose(0, 2, 1)
                            else:
                                cml_attn_result = cml_attn_result

                            passes = np.allclose(
                                torch_result,
                                cml_attn_result,
                                # maximum relative error still gives high values, ~5%, but
                                # likely overlookable from absolute error
                                # rtol=1e-5,
                                atol=5e-6,  # kinda manually fine tuned atol, but still very low
                            )

                            abs_error = np.abs((cml_attn_result - torch_result))
                            rel_error = np.abs(abs_error / torch_result)
                            print(passes, abs_error.max(), rel_error.max())
                            assert passes, (
                                f"Failed for:\n"
                                f"do_pre_norm: {do_pre_norm}\n"
                                f"do_rope: {do_rope}\n"
                                f"do_norm: {do_norm}\n"
                                f"channels_first: {channels_first}\n"
                                f"multi_query_head: {multi_query_head}\n"
                                f"repeat_interleave: {repeat_interleave}\n"
                            )


                            print(
                                (
                                    f"do_pre_norm: {do_pre_norm}\n"
                                    f"do_rope: {do_rope}\n"
                                    f"do_norm: {do_norm}\n"
                                    f"channels_first: {channels_first}\n"
                                    f"multi_query_head: {multi_query_head}\n"
                                    f"repeat_interleave: {repeat_interleave}\n"
                                )
                            )


if __name__ == "__main__":
    main()
