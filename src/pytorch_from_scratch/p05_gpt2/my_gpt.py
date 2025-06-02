import math
import os
from dataclasses import dataclass

import torch
import transformers
from einops import rearrange
from fancy_einsum import einsum
from torch import nn
from torch.nn import functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass(frozen=True)
class GPTConfig:
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    hidden_size: int = 768
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


class UnidirectionalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        # TODO: get p_dropout
        dropout = 0.1
        super().__init__()
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_emb_size = hidden_size // num_heads
        # sub modules
        self.c_attn = nn.Linear(hidden_size, hidden_size * 3)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(p=dropout)  # USEME
        self.resid_dropout = nn.Dropout(p=dropout)  # USEME

    def forward(self, x: torch.Tensor, cache=None):
        batch, tokens, hidden_size = x.shape
        assert hidden_size == self.hidden_size
        # QKV
        x = self.c_attn(x)
        Q, K, V = rearrange(
            x,
            "batch tokens (qkv heads head_emb)" " -> qkv batch heads tokens head_emb",
            qkv=3,
            heads=self.num_heads,
            head_emb=self.head_emb_size,
        )
        # dot [q @ K over head_emb] -> batch heads tokens_q tokens_k
        attn = einsum(
            "batch heads tokens_q head_emb, batch heads tokens_k head_emb"
            " -> batch heads tokens_q tokens_k",
            Q,
            K,
        )
        # scale down
        attn = attn / (self.head_emb_size) ** 0.5
        # mask - if q_idx < k_idx, set to -1e4
        mask = torch.tril(attn.new_ones(size=attn.size())).bool()
        attn = torch.where(mask, attn, -1e4)
        # softmax [over tokens_k] -> batch heads tokens_q tokens_k
        attn_p = F.softmax(attn, dim=-1)
        # weighted sum [attn_p * V]
        out = einsum(
            "batch heads tokens_q tokens_kv, batch heads tokens_kv head_emb"
            " -> batch heads tokens_q head_emb",
            attn_p,
            V,
        )
        # unsplit heads
        out = rearrange(
            out, "batch heads tokens_q head_emb -> batch tokens_q (heads head_emb)"
        )
        # O -> batch tokens hidden_size
        out = self.c_proj(out)
        # output
        return out


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function.

    Currently in Google BERT repo, identical to OpenAI GPT).
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class GPT2MLP(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        # modoles
        self.c_fc = nn.Linear(hidden_size, hidden_size * 4)
        self.c_proj = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.GELU()
        # self.act = NewGELUActivation()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        layer_norm_epsilon: float,
    ):
        super().__init__()
        # modoles
        self.ln_1 = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        self.attn = UnidirectionalAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        self.mlp = GPT2MLP(hidden_size, dropout)

    def forward(self, x: torch.Tensor, cache=None):
        x1 = self.ln_1(x)
        x1 = self.attn(x1)
        x = x1 + x
        x2 = self.ln_2(x)
        x2 = self.mlp(x2)
        x = x2 + x
        return x


class GPT2Transformer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # modules
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(p=config.dropout)
        self.h = nn.Sequential(
            *[
                GPT2Block(
                    config.hidden_size,
                    config.num_heads,
                    config.dropout,
                    config.layer_norm_epsilon,
                )
                for i in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_epsilon)

    def forward(self, x: torch.Tensor, cache=None):
        batch, token = x.shape
        pos = torch.arange(end=token, dtype=int, device=x.device)
        x = self.wte(x) + self.wpe(pos)
        x = self.drop(x)
        x = self.h(x)
        x = self.ln_f(x)
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # modules
        self.transformer = GPT2Transformer(config)
        self.lm_head = nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias=False
        )

    def forward(self, x: torch.Tensor, cache=None):
        x = self.transformer(x)
        # x = self.lm_head(x)
        x = einsum(
            "batch token embedding, vocab embedding -> batch token vocab",
            x,
            self.transformer.wte.weight,
        )
        return x


def load_pretrained_weights():
    pretrained_gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    config = GPTConfig()
    my_gpt = GPT2(config)
    unneeded_keys = set(pretrained_gpt.state_dict()) - set(my_gpt.state_dict())
    missing_keys = set(my_gpt.state_dict()) - set(pretrained_gpt.state_dict())
    assert not missing_keys

    new_state_dict = {}
    for k, v in pretrained_gpt.state_dict().items():
        if k in unneeded_keys:
            continue
        if "c_attn.weight" in k or "c_fc.weight" in k or "c_proj.weight" in k:
            v = v.T
        new_state_dict[k] = v
    my_gpt.load_state_dict(new_state_dict)
    return my_gpt


def sample_next_token(
    model, input_ids: torch.Tensor, temperature=1.0, freq_penalty=2.0, cache=None
) -> int:
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0
    model.eval()
    logits: torch.Tensor = (model(input_ids.unsqueeze(0)))[0][-1].detach()
    if temperature == 0:
        return logits.argmax()
    logits = logits / temperature
    logits.index_put_(
        (input_ids,),
        -torch.tensor(freq_penalty, device=logits.device, dtype=logits.dtype),
        accumulate=True,
    )
    dist = torch.distributions.categorical.Categorical(
        logits=logits, validate_args=None
    )
    return dist.sample()


def sample_tokens(
    model,
    tokenizer: transformers.GPT2Tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=2.0,
    stop_at_eos=True,
    cache=None,
) -> str:
    model.eval()
    input_ids: list[int] = tokenizer(initial_text)["input_ids"]
    for _i_token in range(max_tokens_generated):
        next_token = sample_next_token(
            model,
            torch.tensor(input_ids, dtype=int),
            temperature=temperature,
            freq_penalty=freq_penalty,
            cache=cache,
        )  # .item()
        input_ids.append(next_token)
        if stop_at_eos and next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids)
