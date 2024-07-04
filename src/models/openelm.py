from typing import List
from collections import OrderedDict

import os
import shutil
import numpy as np
from coremltools.converters.mil import Builder as mb, Var

from attention import (
    RoPEEmbedding,
    attention_monstrosity,
    make_rope_mask_indices,
    Mask,
    stateful_attention,
)
from layers import RMSNorm, FFN, Embedding, Head, Conv, Linear


class Attention:
    def __init__(
        self,
        qkvproj: Linear,
        outproj: Linear,
        headdim,
        nqheads,
        nkvheads,
        qnorm: RMSNorm,
        knorm: RMSNorm,
        rope: RoPEEmbedding,
        channels_first,
        pre_normalization_and_pos_encoding,
        multi_query_head,
        repeat_interleave,
        block_index,
        index_in_attention_group: int = -1,
        state_implementation="",
        state_update_at="",
    ):
        self.qkvproj = qkvproj
        self.outproj = outproj
        self.headdim = headdim
        self.nqheads = nqheads
        self.nkvheads = nkvheads
        self.qnorm = qnorm
        self.knorm = knorm
        self.rope = rope
        self.channels_first = channels_first
        self.pre_normalization_and_pos_encoding = pre_normalization_and_pos_encoding
        self.multi_query_head = multi_query_head
        self.repeat_interleave = repeat_interleave
        self.block_index = block_index
        self.index_in_attention_group = index_in_attention_group
        self.state_implementation = state_implementation
        self.state_update_at = state_update_at

    def __call__(
        self,
        hidden_states: Var,
        mask=None,
        query_pos=None,
        split_key_state=None,
        split_value_state=None,
        key_state=None,
        value_state=None,
        query_cos_emb=None,
        query_sin_emb=None,
    ):
        qkv = self.qkvproj(hidden_states, name=f"attention_{self.block_index}_qkvproj")
        # return qkv
        if query_pos is None:
            attention, *_ = attention_monstrosity(
                qkv=qkv,
                mask=mask,
                # mask=None,
                headdim=self.headdim,
                nqheads=self.nqheads,
                nkvheads=self.nkvheads,
                qnorm=self.qnorm,
                knorm=self.knorm,
                rope=self.rope,
                # rope=None,
                channels_first=self.channels_first,
                pre_normalization_and_pos_encoding=self.pre_normalization_and_pos_encoding,
                multi_query_head=self.multi_query_head,
                repeat_interleave=self.repeat_interleave,
                block_index=self.block_index,
            )
        else:
            attention, new_kheads, new_vheads, *_ = stateful_attention(
                qkv=qkv,
                mask=mask,
                query_pos=query_pos,
                split_key_state=split_key_state,
                split_value_state=split_value_state,
                key_state=key_state,
                value_state=value_state,
                headdim=self.headdim,
                nqheads=self.nqheads,
                nkvheads=self.nkvheads,
                update_index=self.index_in_attention_group,
                qnorm=self.qnorm,
                knorm=self.knorm,
                query_sin_emb=query_sin_emb,
                query_cos_emb=query_cos_emb,
                channels_first=self.channels_first,
                block_index=self.block_index,
                state_implementation=self.state_implementation,
                state_update_at=self.state_update_at,
            )
        out = self.outproj(attention, name=f"attention_{self.block_index}_outproj")
        return out, new_kheads, new_vheads


class Block:
    def __init__(
        self,
        attn_norm: RMSNorm,
        attn: Attention,
        ffn_norm: RMSNorm,
        ffn: FFN,
        # axis: int,
        channels_first: bool,
        block_index: int,
    ):
        self.attn_norm = attn_norm
        self.attn = attn
        self.ffn_norm = ffn_norm
        self.ffn = ffn
        # self.axis = axis
        self.channels_first = channels_first
        self.block_index = block_index

    def __call__(
        self,
        hidden_states: Var,
        mask=None,
        # axis=None,
        channels_first=None,
        query_pos=None,
        split_key_state=None,
        split_value_state=None,
        key_state=None,
        value_state=None,
        query_sin_emb=None,
        query_cos_emb=None,
    ):
        # if axis is None:
        #     axis = self.axis
        axis = 1 if channels_first else 2

        residual = hidden_states
        hidden_states = self.attn_norm(
            hidden_states,
            prefix=f"block_{self.block_index}_attention",
            axes=[axis],
        )
        hidden_states = self.attn(
            hidden_states,
            mask=mask,
            query_pos=query_pos,
            split_key_state=split_key_state,
            split_value_state=split_value_state,
            key_state=key_state,
            value_state=value_state,
            query_cos_emb=query_cos_emb,
            query_sin_emb=query_sin_emb,
        )
        if query_pos is not None:
            hidden_states, new_kheads, new_vheads = hidden_states
        hidden_states = mb.add(
            x=residual, y=hidden_states, name=f"block_{self.block_index}_residual_1"
        )
        residual = hidden_states
        hidden_states = self.ffn_norm(
            hidden_states, prefix=f"block_{self.block_index}_ffn", axes=[axis]
        )
        hidden_states = self.ffn(
            hidden_states, prefix=f"block_{self.block_index}", axis=axis
        )
        hidden_states = mb.add(
            x=hidden_states, y=residual, name=f"block_{self.block_index}_residual_2"
        )
        if query_sin_emb is not None:
            return hidden_states, new_kheads, new_vheads
        return hidden_states


class OpenELM:
    def __init__(
        self,
        embedding: Embedding,
        head: Head,
        blocks: List[Block],
        finalnorm: RMSNorm,
        rope: RoPEEmbedding,
        mask: Mask,
        channels_first=True,
        attention_groups=OrderedDict(),
        state_implementation="",
        state_update_at="",
    ):
        self.embedding = embedding
        self.head = head
        self.blocks = blocks
        self.finalnorm = finalnorm
        self.rope = rope
        self.mask = mask
        self.channels_first = channels_first
        self.attention_groups = attention_groups

        self.state_implementation = state_implementation
        self.state_update_at = state_update_at

    def __call__(
        self,
        input_ids,
        query_pos=None,
        mask=None,
        query_sin_emb=None,
        query_cos_emb=None,
        # key_state=None,
        # value_state=None,
        states=None,
        # hidden_states,
        channels_first=None,
        num_blocks=-1,
    ):
        if channels_first is None:
            channels_first = self.channels_first
        if channels_first:
            axis = 1
        else:
            axis = 2
        # little hack since we have hidden_states that contain feature dimension which we do not want
        # NOTE: For some reason this does not build to CoreML
        # indices = mb.gather(x=hidden_states, indices=0, axis=axis)

        # indices = make_rope_mask_indices(indices)
        if states is None:
            indices = make_rope_mask_indices(input_ids)
            mask = self.mask.get_mask(indices)
            self.rope.init_embedding_slices(indices, self.channels_first)

        hidden_states = self.embedding(input_ids)
        if channels_first:
            hidden_states = mb.transpose(
                x=hidden_states, perm=[0, 2, 1], name="input_embeddings_channels_first"
            )

        if self.state_implementation.startswith("per_group"):
            key_states = {}
            value_states = {}
            split_keys = {}
            split_values = {}
            added_blocks = 0
            if states is not None:
                for i, (num_heads, num_layers) in enumerate(
                    list(self.attention_groups.items())  # [:num_blocks]
                ):
                    _ks = states[i * 2]
                    _vs = states[i * 2 + 1]
                    _rks = mb.read_state(input=_ks)
                    _rvs = mb.read_state(input=_vs)
                    key_states[num_heads] = (_ks, _rks)
                    value_states[num_heads] = (_vs, _rvs)
                    split_keys[num_heads] = mb.split(
                        x=_rks,
                        split_sizes=[num_heads] * num_layers,
                        axis=1,
                        name=f"key_state_{num_heads}",
                    )
                    split_values[num_heads] = mb.split(
                        x=_rvs,
                        split_sizes=[num_heads] * num_layers,
                        axis=1,
                        name=f"value_state_{num_heads}",
                    )
                    added_blocks += num_layers
                    if added_blocks >= num_blocks:
                        break
        elif self.state_implementation.startswith("big_state"):
            key_states, value_states = states
            key_states_read, value_states_read = mb.read_state(
                input=key_states
            ), mb.read_state(input=value_states)
            # maybe try in the future
            # if self.state_implementation == "big_state_split":
            if state_update_at.startswith("attention"):
                pass
            elif state_update_at == "block":
                pass
            elif state_update_at == "group":
                pass
            elif state_update_at == "end":
                pass
        elif self.state_implementation == "per_block":
            pass
        new_kheads, new_vheads = [], []
        group_i = 0
        block: Block
        ops_at_end = []  # to prevent ANE graph break
        total_num_heads = 0
        current_num_heads = 0
        state_counter = 0
        for i, block in enumerate(self.blocks[:num_blocks]):
            nkvheads = block.attn.nkvheads
            group_size = self.attention_groups[nkvheads]

            if self.state_implementation == "big_state_gather":
                split_key_state = mb.gather(
                    x=key_states_read,
                    axis=1,
                    indices=[total_num_heads + j for j in range(nkvheads)],
                )
                split_value_state = mb.gather(
                    x=value_states_read,
                    axis=1,
                    indices=[total_num_heads + j for j in range(nkvheads)],
                )
                key_state = (key_states, key_states_read)
                value_state = (value_states, value_states_read)
            elif self.state_implementation == "big_state_slice":
                split_key_state = mb.slice_by_size(
                    x=key_states_read,
                    begin=[0, total_num_heads, 0, 0],
                    size=[-1, nkvheads, -1, -1],
                )
                split_value_state = mb.slice_by_size(
                    x=value_states_read,
                    begin=[0, total_num_heads, 0, 0],
                    size=[-1, nkvheads, -1, -1],
                )
                key_state = (key_states, key_states_read)
                value_state = (value_states, value_states_read)
            elif self.state_implementation.startswith("per_group"):

                if group_i == 0:
                    key_state = key_states[nkvheads]
                    value_state = value_states[nkvheads]
                    key_state_read = mb.read_state(input=key_state)
                    value_state_read = mb.read_state(input=value_state)
                    key_state = (key_state, key_state_read)
                    value_state = (value_state, value_state_read)

                    if self.state_implementation == "per_group_split":
                        key_state_splits = mb.split(
                            x=key_state_read,
                            axis=1,
                            split_sizes=[nkvheads] * group_size,
                        )
                        value_state_splits = mb.split(
                            x=value_state_read,
                            axis=1,
                            split_sizes=[nkvheads] * group_size,
                        )

                if self.state_implementation == "per_group_split":
                    split_key_state = key_state_splits[group_i]
                    split_value_state = value_state_splits[group_i]
                elif self.state_implementation == "per_group_slice":
                    split_key_state = mb.slice_per_size(
                        x=key_state_read,
                        begin=[0, group_i * nkvheads, 0, 0],
                        end=[-1, group_i * (nkvheads + 1), -1, -1],
                    )
                    split_value_state = mb.slice_per_size(
                        x=value_state_read,
                        begin=[0, group_i * nkvheads, 0, 0],
                        end=[-1, group_i * (nkvheads + 1), -1, -1],
                    )
                elif self.state_implementation == "per_group_gather":
                    split_key_state = mb.gather(
                        x=key_states_read,
                        axis=1,
                        indices=[group_i + j for j in range(nkvheads)],
                    )
                    split_value_state = mb.gather(
                        x=value_states_read,
                        axis=1,
                        indices=[group_i + j for j in range(nkvheads)],
                    )
            elif self.state_implementation == "per_block":
                key_state = (states[2 * i],) # tuple for consistency
                value_state = (states[2 * i + 1],)
                # key_state_read = mb.read_state(input=key_state)
                # value_state_read = mb.read_state(input=value_state)

                # key_state_read = mb.transpose(
                #     x=key_state_read, perm=[0, 2, 3, 1],
                # )
                # value_state_read = mb.transpose(
                #     x=value_state_read, perm=[0, 2, 3, 1],
                # )

                # key_state = (key_state, key_state_read)
                # value_state = (value_state, value_state_read)

                split_key_state = None
                split_value_state = None
            elif self.state_implementation == "max_num_heads":
                if current_num_heads == 0:
                    key_state = states[2 * state_counter]
                    value_state = states[2 * state_counter + 1]
                    key_state_read = mb.read_state(input=key_state)
                    value_state_read = mb.read_state(input=value_state)
                
                split_key_state = mb.slice_by_size(
                    x=key_state_read,
                    begin=[0, current_num_heads, 0, 0],
                    size=[-1, nkvheads, -1, -1],
                )
                split_value_state = mb.slice_by_size(
                    x=value_state_read,
                    begin=[0, current_num_heads, 0, 0],
                    size=[-1, nkvheads, -1, -1],
                )

                current_num_heads += nkvheads


            total_num_heads += nkvheads

            hidden_states = block(
                hidden_states,
                mask=mask,
                channels_first=channels_first,
                query_pos=query_pos,
                # split_key_state=split_keys[block.attn.nkvheads][
                #     block.attn.index_in_attention_group
                # ],
                # split_value_state=split_values[block.attn.nkvheads][
                #     block.attn.index_in_attention_group
                # ],
                split_key_state=split_key_state,
                split_value_state=split_value_state,
                # key_state=key_states[block.attn.nkvheads],
                # value_state=value_states[block.attn.nkvheads],
                key_state=key_state,
                value_state=value_state,
                query_cos_emb=query_cos_emb,
                query_sin_emb=query_sin_emb,
            )
            if query_pos is not None:
                hidden_states, _new_kheads, _new_vheads = hidden_states
                new_kheads.append(_new_kheads)
                new_vheads.append(_new_vheads)

            if self.state_update_at.startswith("attention"):
                pass
            elif self.state_update_at == "block":
                if self.state_implementation == "big_state_gather":
                    # we have to perform an slice update in which
                    # token position is dynamic but heads are fixed
                    # it can be performed using the pre-scattered
                    # new_kv of the attention across all positions,
                    # or using the single new kv only on query_pos
                    # (or maybe a scatter over multiple axes, which i don't know how
                    # to do)

                    ## Single new position
                    pass

                elif self.state_implementation == "big_state_slice":
                    pass
                elif self.state_implementation == "per_group_split":
                    pass
                elif self.state_implementation == "per_group_splice":
                    pass
                elif self.state_implementation == "per_group_gather":
                    pass
                elif self.state_implementation == "per_block":
                    pass

            elif self.state_update_at == "group":
                if self.state_implementation == "max_num_heads":
                    if current_num_heads >= key_state_read.shape[1]:
                        new_kheads = mb.concat(values=new_kheads, axis=1)
                        new_vheads = mb.concat(values=new_vheads, axis=1)

                        updated_key_state = mb.scatter(
                            data=key_state_read,
                            updates=new_kheads,
                            axis=2,
                            indices=query_pos,
                        )
                        updated_value_state = mb.scatter(
                            data=value_state_read,
                            updates=new_vheads,
                            axis=2,
                            indices=query_pos,
                        )

                        mb.coreml_update_state(state=key_state, value=updated_key_state)
                        mb.coreml_update_state(state=value_state, value=updated_value_state)
                        
                        new_kheads = []
                        new_vheads = []
                        current_num_heads = 0
                        state_counter += 1
            elif self.state_update_at == "end":
                if self.state_implementation == "per_block":
                    ops_at_end.append(
                        {
                            "name": "update_state",
                            "updates": _new_kheads,
                            # _new_kheads is in (batch, head, seqlen, headdim)
                            # convert o (batch, headdim, head, seqlen)
                            # "updates": mb.transpose(x=_new_kheads, perm=[0, 3, 1, 2]),
                            "state": key_state[0],
                        }
                    )
                    ops_at_end.append(
                        {
                            "name": "update_state",
                            # "updates": mb.transpose(x=_new_vheads, perm=[0, 3, 1, 2]),
                            "updates": _new_vheads,
                            "state": value_state[0],
                        }
                    )
                

            group_i += 1
            if group_size == group_i:
                group_i = 0

        hidden_states = self.finalnorm(hidden_states, axes=[axis], prefix="final_norm")
        out = self.head(hidden_states)

        if (
            self.state_implementation.startswith("big_state")
            and self.state_update_at == "end"
        ):
            new_kheads = mb.concat(values=new_kheads, axis=1)
            new_vheads = mb.concat(values=new_vheads, axis=1)

            updated_key_state = mb.scatter(
                data=key_states_read,
                updates=new_kheads,
                axis=2,
                indices=query_pos,
            )
            updated_value_state = mb.scatter(
                data=value_states_read,
                updates=new_vheads,
                axis=2,
                indices=query_pos,
            )

            mb.coreml_update_state(state=key_states, value=updated_key_state)
            mb.coreml_update_state(state=value_states, value=updated_value_state)

        else:
            for op in ops_at_end[:24]:
                if op["name"] == "scatter":
                    updated_var = mb.scatter(
                        data=op["x"],
                        updates=op["updates"],
                        axis=op["axis"],
                        indices=op["indices"],
                    )
                    mb.coreml_update_state(state=op["state"], value=updated_var)

                elif op["name"] == "update_state":
                    mb.coreml_update_state(state=op["state"], value=op["updates"])

        # updated_key_state = mb.scatter(
        #     data=key_states_read,
        #     updates=new_kheads,
        #     indices=query_pos,
        #     axis=2,
        # )
        # updated_value_state = mb.scatter(
        #     data=value_states_read,
        #     updates=new_vheads,
        #     indices=query_pos,
        #     axis=2,
        # )

        # begin = np.array([0, 0, 0, 0])
        # end = np.array([-1, 63, 1, -1])
        # add = np.array([0, 0, 1, 0])
        # update_mask = [True, False, False, True]
        # add = mb.mul(x=add, y=query_pos)
        # begin = mb.add(x=begin, y=add)
        # end = mb.add(x=end, y=add)

        # updated_key_state = mb.slice_update(
        #     x=key_states_read,
        #     update=new_kheads,
        #     begin=begin,
        #     end=end,
        #     begin_mask=update_mask,
        #     end_mask=update_mask,
        # )
        # updated_value_state = mb.slice_update(
        #     x=value_states_read,
        #     update=new_vheads,
        #     begin=begin,
        #     end=end,
        #     begin_mask=update_mask,
        #     end_mask=update_mask,
        # )

        # mb.coreml_update_state(state=key_states, value=updated_key_state)
        # mb.coreml_update_state(state=value_states, value=updated_value_state)

        return out


def convert_linear(layer, name=None, channels_first=True, dtype=np.float32):
    weight = layer.weight.detach().numpy().astype(dtype)
    if channels_first:
        weight = np.expand_dims(weight, -1)
        return Conv(weight, name=name)
    else:
        return Linear(weight, name=name)


def convert_rmsnorm(layer, name, channels_first=True, dtype=np.float32):
    weight = layer.weight.detach().numpy().astype(dtype)
    if channels_first:
        axis = 1
        weight = np.expand_dims(weight, -1)
    else:
        axis = -1
    return RMSNorm(weight, layer.eps, axes=[axis], name=name)


def convert_ffn(layer, block_index, name, channels_first=True, dtype=np.float32):
    linear1 = convert_linear(
        layer.proj_1,
        channels_first=channels_first,
        name=f"block_{block_index}_ffn_in_proj",
        dtype=dtype,
    )
    linear2 = convert_linear(
        layer.proj_2,
        channels_first=channels_first,
        name=f"block_{block_index}_ffn_out_proj",
        dtype=dtype,
    )
    # TEMP hardcode activation
    activation = mb.silu
    if channels_first:
        axis = 1
    else:
        axis = -1

    return FFN(linear1, linear2, activation, layer.ffn_with_glu, block_index, axis=axis)


def from_torch(
    pytorch_model,
    channels_first,
    pre_normalization_and_pos_encoding,
    multi_query_head,
    repeat_interleave,
    dtype=np.float32,
    state_implementation="",
    state_update_at="",
):
    axis = 1 if channels_first else 2
    blocks: List[Block] = []
    attention_groups = OrderedDict()
    # ropes = []
    rope = RoPEEmbedding(64, 2048, 2048, channels_first=channels_first, dtype=dtype)
    for i, layer in enumerate(pytorch_model.transformer.layers):
        attn = layer.attn
        hdim = layer.attn.head_dim

        attn_norm = convert_rmsnorm(
            layer.attn_norm, f"layer_{i}_attention_rmsnorm", channels_first, dtype
        )
        qnorm = convert_rmsnorm(
            layer.attn.q_norm, f"layer_{i}_attention_q_rmsnorm", channels_first, dtype
        )
        knorm = convert_rmsnorm(
            layer.attn.k_norm, f"layer_{i}_attention_k_rmsnorm", channels_first, dtype
        )

        # q, k, v = np.split(
        #     attn.qkv_proj.weight.detach().numpy(),
        #     indices_or_sections=[
        #         attn.head_dim * attn.num_q_heads,
        #         attn.head_dim * attn.num_q_heads + attn.head_dim * attn.num_k_heads,
        #         # attn.head_dim * attn.num_v_heads,
        #     ],
        #     axis=0,
        # )

        # qproj = Linear(q)
        # kproj = Linear(k)
        # vproj = Linear(v)
        qkvproj = convert_linear(
            attn.qkv_proj, f"block_{i}_attention_qkvproj", channels_first, dtype
        )
        outproj = convert_linear(
            layer.attn.out_proj, f"attention_{i}_outproj", channels_first, dtype
        )
        # if hdim not in ropes:
        #     ropes[hdim] = RoPEEmbedding(hdim, 2048, 2048, dtype=dtype)
        if layer.attn.num_k_heads not in attention_groups:
            attention_groups[layer.attn.num_k_heads] = 0
        new_attn = Attention(
            qkvproj,
            outproj,
            hdim,
            layer.attn.num_q_heads,
            layer.attn.num_k_heads,
            qnorm,
            knorm,
            rope,
            channels_first,
            pre_normalization_and_pos_encoding,
            multi_query_head,
            repeat_interleave,
            block_index=i,
            index_in_attention_group=attention_groups[layer.attn.num_k_heads],
            state_implementation=state_implementation,
            state_update_at=state_update_at,
        )
        attention_groups[layer.attn.num_k_heads] += 1
        ffn = convert_ffn(layer.ffn, i, f"layer_{i}_ffn", channels_first, dtype)
        ffn_norm = convert_rmsnorm(
            layer.ffn_norm, f"layer_{i}_ffn_rmsnorm", channels_first, dtype
        )

        block = Block(
            attn_norm,
            new_attn,
            ffn_norm,
            ffn,
            channels_first,
            i,
        )

        blocks.append(block)

    embs = (
        pytorch_model.transformer.token_embeddings.weight.detach().numpy().astype(dtype)
    )

    tokemb = Embedding(
        embs,
        name="token_embedding",
    )
    head = Head(
        embs,
        # nsplits=2,
        channels_first=channels_first,
        split_size=16_000,
        prefix="lm_head",
    )
    finalnorm = convert_rmsnorm(
        pytorch_model.transformer.norm, "final_rmsnorm", channels_first, dtype
    )

    model = OpenELM(
        tokemb,
        head,
        blocks,
        finalnorm,
        rope,
        Mask(2048, dtype),
        channels_first,
        attention_groups=attention_groups,
        state_implementation=state_implementation,
        state_update_at=state_update_at,
    )

    return model


if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn

    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import coremltools.converters.mil as mil
    from transformers import AutoModelForCausalLM

    torch_model = AutoModelForCausalLM.from_pretrained(
        "apple/OpenELM-270M", trust_remote_code=True
    )

    precision = "fp16"

    if precision == "fp32":
        mtype = mil.input_types.types.fp32
        compute_precision = ct.precision.FLOAT32
        nptype = np.float32
    elif precision == "fp16":
        mtype = mil.input_types.types.fp16
        compute_precision = ct.precision.FLOAT16
        nptype = np.float16
    else:
        mtype = None
        compute_precision = None
        nptype = None

    channels_first = True
    pre_normalization_and_pos_encoding = False
    multi_query_head = False
    repeat_interleave = False

    state_implementation = "per_block" # "per_block", "per_group_split", "per_group_slice", "per_group_gather", "big_state_gather", "big_state_slice", "max_num_heads"
    max_num_heads = 19
    state_update_at = "attention"
    seqlength = 512

    coreml_model = from_torch(
        torch_model,
        channels_first=channels_first,
        pre_normalization_and_pos_encoding=pre_normalization_and_pos_encoding,
        multi_query_head=multi_query_head,
        repeat_interleave=repeat_interleave,
        dtype=nptype,
        state_implementation=state_implementation,
        state_update_at=state_update_at,
    )

    shift = 0
    num_blocks = -1
    coreml_model.blocks = coreml_model.blocks[shift:]  # for testing/debugging stuff
    if num_blocks == -1:
        num_blocks = len(coreml_model.blocks)
    coreml_model.blocks = coreml_model.blocks[:num_blocks]

    # shapes = [(1, 1280, seqlen) for seqlen in (32, 128, 256, 512, 1024, 2048)]
    shapes = [(1, seqlen) for seqlen in (32, 128, 256, 512, 1024, 2048)]
    enum_shape = mil.input_types.EnumeratedShapes(shapes=shapes)

    # fixed_shape = (1, 1280, 128)
    # fixed_shape = (1, 128)
    fixed_shape = (1, 1)
    # shape = enum_shape.symbolic_shape
    # shape = fixed_shape

    # num_groups = 0
    # added_blocks = 0
    # for nblocks in coreml_model.attention_groups.values():
    #     added_blocks += nblocks
    #     num_groups += 1
    #     if added_blocks >= num_blocks:
    #         break

    block: Block
    total_num_heads = 0
    groups = OrderedDict()
    for block in coreml_model.blocks:
        nkvheads = block.attn.nkvheads
        if nkvheads not in groups:
            groups[nkvheads] = 0
        groups[nkvheads] += 1
        total_num_heads += nkvheads
    num_groups = len(groups)

    if state_implementation == "per_block":
        num_states = num_blocks

        state_spec = sum(
            [
                # [
                #     mb.StateTensorSpec(
                #         (1, 64, block.attn.nkvheads, seqlength),
                #         dtype=mil.input_types.types.fp16,
                #     ),
                #     mb.StateTensorSpec(
                #         (1, 64, block.attn.nkvheads, seqlength),
                #         dtype=mil.input_types.types.fp16,
                #     ),
                # ]
                [
                    mb.StateTensorSpec(
                        (1, block.attn.nkvheads, seqlength, 64),
                        dtype=mil.input_types.types.fp16,
                    ),
                    mb.StateTensorSpec(
                        (1, block.attn.nkvheads, seqlength, 64),
                        dtype=mil.input_types.types.fp16,
                    ),
                ]
                for block in coreml_model.blocks
            ],
            [],
        )
    elif state_implementation.startswith("big_state"):
        num_states = 1
        state_spec = [
            mb.StateTensorSpec(
                (1, total_num_heads, seqlength, 64),
                dtype=mil.input_types.types.fp16,
            ),
            mb.StateTensorSpec(
                (1, total_num_heads, seqlength, 64),
                dtype=mil.input_types.types.fp16,
            ),
        ]
    elif state_implementation == "max_num_heads":
        state_spec = []
        current_num_heads = 0
        for i, block in enumerate(coreml_model.blocks):
            if ((block.attn.nkvheads + current_num_heads) > max_num_heads):
                state_spec += [
                    mb.StateTensorSpec(
                        (1, current_num_heads, seqlength, 64),
                        dtype=mil.input_types.types.fp16,
                    ),
                    mb.StateTensorSpec(
                        (1, current_num_heads, seqlength, 64),
                        dtype=mil.input_types.types.fp16,
                    ),
                ]
                current_num_heads = block.attn.nkvheads
            else:
                current_num_heads += block.attn.nkvheads

        if current_num_heads > 0:
            state_spec += [
                mb.StateTensorSpec(
                    (1, current_num_heads, seqlength, 64),
                    dtype=mil.input_types.types.fp16,
                ),
                mb.StateTensorSpec(
                    (1, current_num_heads, seqlength, 64),
                    dtype=mil.input_types.types.fp16,
                ),
            ]

        num_states = len(state_spec) // 2
    # awfull fix to programatically create variable number of arguments and their name
    args_str = ",\n    ".join(
        [f"key_state_{i},\n    value_state_{i}" for i in range(num_states)]
    )
    func_def = f"""
def var_program(
    input_ids,
    query_pos,
    mask,
    query_sin_emb,
    query_cos_emb,
    {args_str}
):
    states = [\n    {args_str}\n    ]
    return coreml_model(
        input_ids=input_ids,
        query_pos=query_pos,
        mask=mask,
        query_sin_emb=query_sin_emb,
        query_cos_emb=query_cos_emb,
        states=states,
        num_blocks={num_blocks},
    )
"""

    print(func_def)

    local_namespace = {}

    # Execute the function definition in the new namespace
    exec(func_def, globals(), local_namespace)

    program_func = local_namespace["var_program"]
    print(program_func)

    coreml_model_program = mb.program(
        input_specs=[
            # mb.TensorSpec(enum_shape.symbolic_shape, dtype=mil.input_types.types.int32),
            mb.TensorSpec(fixed_shape, dtype=mil.input_types.types.int32),
            # mb.TensorSpec(enum_shape.symbolic_shape, dtype=mil.input_types.types.fp16),
            mb.TensorSpec((1,), dtype=mil.input_types.types.int32),  # query_pos
            mb.TensorSpec((1, 1, seqlength), dtype=mil.input_types.types.fp16),  # mask
            mb.TensorSpec(
                (1, 1, 64), dtype=mil.input_types.types.fp16
            ),  # query_sin_emb
            mb.TensorSpec(
                (1, 1, 64), dtype=mil.input_types.types.fp16
            ),  # query_cos_emb
            *state_spec,
        ],
        # opset_version=mil.builder.AvailableTarget.iOS17,
        opset_version=mil.builder.AvailableTarget.iOS18,
    )(program_func)

    # @mb.program(
    #     input_specs=[
    #         # mb.TensorSpec(enum_shape.symbolic_shape, dtype=mil.input_types.types.int32),
    #         mb.TensorSpec(fixed_shape, dtype=mil.input_types.types.int32),
    #         # mb.TensorSpec(enum_shape.symbolic_shape, dtype=mil.input_types.types.fp16),
    #         mb.TensorSpec((1, 1, 512), dtype=mil.input_types.types.fp16),  # mask
    #         mb.TensorSpec((1,), dtype=mil.input_types.types.int32),  # query_pos
    #         mb.TensorSpec(
    #             (1, 1, 64), dtype=mil.input_types.types.fp16
    #         ),  # query_sin_emb
    #         mb.TensorSpec(
    #             (1, 1, 64), dtype=mil.input_types.types.fp16
    #         ),  # query_cos_emb
    #         mb.StateTensorSpec(
    #             (1, 3, 512, 64), dtype=mil.input_types.types.fp16
    #         ),  # key_state
    #         mb.StateTensorSpec(
    #             (1, 3, 512, 64), dtype=mil.input_types.types.fp16
    #         ),  # value_state
    #         mb.StateTensorSpec(
    #             (1, 3, 512, 64), dtype=mil.input_types.types.fp16
    #         ),  # key_state
    #         mb.StateTensorSpec(
    #             (1, 3, 512, 64), dtype=mil.input_types.types.fp16
    #         ),  # value_state
    #     ],
    #     # opset_version=mil.builder.AvailableTarget.iOS17,
    #     opset_version=mil.builder.AvailableTarget.iOS18,
    # )
    # # def coreml_model_program(input_ids):
    # # return coreml_model(input_ids, num_blocks=2)
    # def coreml_model_program(
    #     input_ids,
    #     mask,
    #     query_pos,
    #     query_sin_emb,
    #     query_cos_emb,
    #     # *states,
    #     key_state1,
    #     value_state1,
    #     key_state2,
    #     value_state2,
    # ):
    #     states = [key_state1, value_state1, key_state2, value_state2]
    #     return coreml_model(
    #         input_ids=input_ids,
    #         mask=mask,
    #         query_pos=query_pos,
    #         query_sin_emb=query_sin_emb,
    #         query_cos_emb=query_cos_emb,
    #         # key_state=key_state1,
    #         # value_state=value_state1,
    #         # key_state=key_state2,
    #         # value_state=value_state2,
    #         states=states,
    #         num_blocks=2,
    #     )

    # def coreml_model_progra(hidden_states):
    #     return coreml_model(hidden_states, num_blocks=-1)

    print(coreml_model_program)

    # pipeline = ct.PassPipeline.DEFAULT
    # pipeline.remove_passes({"common::add_int16_cast"})
    cml_converted = ct.convert(
        coreml_model_program,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_precision=ct.precision.FLOAT16,
        compute_precision=compute_precision,
        # minimum_deployment_target=ct.target.iOS17,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            # ct.TensorType(name="input_ids", shape=ct.EnumeratedShapes(shapes)),
            ct.TensorType(name="input_ids", shape=fixed_shape),
            # ct.TensorType(name="hidden_states", shape=ct.EnumeratedShapes(shapes)),
            # ct.TensorType(name="input_ids", shape=shape),
            ct.TensorType(name="query_pos", shape=(1,)),
            ct.TensorType(name="mask", shape=(1, 1, seqlength)),
            ct.TensorType(name="query_sin_emb", shape=(1, 1, 64)),
            ct.TensorType(name="query_cos_emb", shape=(1, 1, 64)),
        ],
        # pass_pipeline=pipeline,
    )

    print("aaah")

    name = f"openelm_270m_fp16_stateful_attention_shift_{shift}_{state_implementation}_{state_update_at}_{num_blocks}_{seqlength}.mlpackage"
    if os.path.exists(name):
        shutil.rmtree(name)

    try:
        state = cml_converted.make_state()
        cml_converted.predict(
            {
                "input_ids": np.array([[0]], dtype=np.int32),
                "query_pos": np.array([0], dtype=np.int32),
                "mask": np.zeros((1, 1, seqlength), dtype=np.float16),
                "query_sin_emb": np.random.randn(1, 1, 64).astype(np.float16),
                "query_cos_emb": np.random.randn(1, 1, 64).astype(np.float16),
            },
            state,
        )
        cml_converted.save(name)
    except Exception as e:
        print(e)
        cml_converted.save(f"F_{name}")

    try:
        print(cml_converted._get_mil_internal())
    except Exception as e:
        print(e)
