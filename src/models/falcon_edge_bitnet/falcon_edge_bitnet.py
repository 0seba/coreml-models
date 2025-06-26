from typing import List, Optional


from models.llama.llama import (
    LlamaRMSNorm,
    LlamaAttentionLayer,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
)


class FalconEdgeRMSNorm(LlamaRMSNorm):
    """"""


class FalconEdgeAttentionLayer(LlamaAttentionLayer):
    """"""


class FalconEdgeMLP(LlamaMLP):
    """"""


class FalconEdgeBitnetDecoderLayer(LlamaDecoderLayer):  # Subclass LlamaDecoderLayer
    def __init__(
        self,
        input_layernorm: FalconEdgeRMSNorm,  # Override type hints
        attention: FalconEdgeAttentionLayer,
        post_attention_layernorm: FalconEdgeRMSNorm,
        ffn: FalconEdgeMLP,
    ):
        super().__init__(
            input_layernorm, attention, post_attention_layernorm, ffn
        )  # Call superclass __init__


class FalconEdgeBitnetModel(LlamaModel):
    def __init__(
        self,
        blocks: List[FalconEdgeBitnetDecoderLayer],
        max_sequence_length=8192,
        sin_emb: Optional = None,
        cos_emb: Optional = None,
        layer_from: int = 0,
    ):
        super().__init__(blocks, max_sequence_length, sin_emb, cos_emb, layer_from)
