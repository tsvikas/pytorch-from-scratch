from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn
from torch.nn import Embedding, LayerNorm  # can also import from .my_nn_modules


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int = 28996
    intermediate_size: int = 3072
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_position_embeddings: int = 512
    dropout: float = 0.1
    type_vocab_size: int = 2  # unused
    layer_norm_epsilon: float = 1e-12


@dataclass
class BertOutput:
    logits: None | torch.Tensor = None
    is_positive: None | torch.Tensor = None
    star_rating: None | torch.Tensor = None


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_heads
        assert config.hidden_size % config.num_heads == 0
        self.qk_embedding = qk_embedding = config.hidden_size // config.num_heads
        v_embedding = qk_embedding
        self.project_query = nn.Linear(
            config.hidden_size, config.num_heads * qk_embedding
        )
        self.project_key = nn.Linear(
            config.hidden_size, config.num_heads * qk_embedding
        )
        self.project_value = nn.Linear(
            config.hidden_size, config.num_heads * v_embedding
        )
        self.project_output = nn.Linear(
            config.num_heads * v_embedding, config.hidden_size
        )

    def attention_pattern_pre_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """
        # x:         batch_size, token, embedding
        # q_project: batch_size, token, head * qk_embedding
        # k_project: batch_size, token, head * qk_embedding
        # return:    batch_size, head, q, k
        q = rearrange(
            self.project_query(x),
            "batch q (head qk_embedding) -> batch head q qk_embedding",
            head=self.num_heads,
        )
        k = rearrange(
            self.project_key(x),
            "batch k (head qk_embedding) -> batch head k qk_embedding",
            head=self.num_heads,
        )
        attention = einsum(
            "batch head q qk_embedding, batch head k qk_embedding -> batch head q k",
            q,
            k,
        )
        return attention / (self.qk_embedding**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:         batch_size, token, embedding
        # return:    batch_size, token, embedding

        # softmax over k
        attention = self.attention_pattern_pre_softmax(x).softmax(dim=-1)
        v = rearrange(
            self.project_value(x),
            "batch token (head v_embedding) -> batch head token v_embedding",
            head=self.num_heads,
        )
        weighted_sum = einsum(
            "batch head k v_embedding, batch head q k -> batch head q v_embedding",
            v,
            attention,
        )
        weighted_sum = rearrange(
            weighted_sum, "batch head q v_embedding -> batch q (head v_embedding)"
        )
        output = self.project_output(weighted_sum)
        assert output.shape == x.shape
        return output


class BertMLP(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = torch.nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.linear1(x)
        left = self.gelu(left)
        left = self.linear2(left)
        left = self.dropout(left)
        output = self.layer_norm(left + x)
        return output


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.self_attention(x)
        left = self.dropout(left)
        output = self.layer_norm(left + x)
        return output


class BertBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.mlp = BertMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.mlp(x)
        return x


class Unembed(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.unembed_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.unembed_bias


class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.type_embedding = Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout)
        self.bert_blocks = nn.Sequential(
            *[BertBlock(config) for i in range(config.num_layers)]
        )
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.unembed = Unembed(config)

    def forward(self, input_ids, token_type_ids=None) -> BertOutput:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)
        position = repeat(
            torch.arange(input_ids.shape[1]),
            "pos -> batch pos",
            batch=input_ids.shape[0],
        )
        out = (
            self.token_embedding(input_ids)
            + self.position_embedding(position)
            + self.type_embedding(token_type_ids)
        )
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.bert_blocks(out)
        out = self.linear(out)
        out = self.gelu(out)
        out = self.layer_norm2(out)
        # tied unembed
        out = einsum(
            "batch token hidden, vocab hidden-> batch token vocab",
            out,
            self.token_embedding.weight,
        )
        out = self.unembed(out)
        return BertOutput(logits=out)
