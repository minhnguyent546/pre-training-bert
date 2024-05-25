"""Bert model

Bert implementation from scratch, following the Bert's paper:
Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding,
and the original implementation from google: https://github.com/google-research/bert

requires: python >= 3.10
"""

from dataclasses import dataclass
import math

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F


@dataclass
class BertConfig:
    vocab_size: int = 32_000
    type_vocab_size: int = 2  # used for token_type_ids, or 16 if we need more token types
    max_masked_tokens: int = 10
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_seq_length: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1
    norm_eps: float = 1e-7
    activation: str = 'gelu'


class LayerNorm(nn.Module):
    def __init__(self, features, eps: float = 1e-7):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std
        output = self.weight * y + self.bias
        return output

class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        seq_length = input_ids.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.int32, device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32, device=input_ids.device)

        # token embedding
        embeddings = self.token_embeddings(input_ids)

        # add positional embeddings and token type embeddings, then apply
        # layer normalization and dropout
        embeddings += self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if not config.hidden_size % config.num_heads == 0:
            raise ValueError(
                f'The hidden size {config.hidden_size} is not divisible by '
                f'the number of attention heads {config.num_heads}'
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.d_k = config.hidden_size // config.num_heads
        self.dropout = nn.Dropout(config.dropout)
        self.attention_dropout = nn.Dropout(config.attn_dropout)
        self.w_qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.w_o = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_length, _ = x.size()

        q, k, v = self.w_qkv(x).split(self.hidden_size, dim=-1)

        # q, k, v: (batch_size, seq_length, hidden_size) -> (batch_size, num_heads, seq_length, d_k)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        y = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                         dropout=self.attention_dropout)
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        y = self.w_o(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = get_activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x

class BertEncoderLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = LayerNorm(config.hidden_size, config.norm_eps)
        self.ff_norm = LayerNorm(config.hidden_size, config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x = self.attention_norm(x + self.dropout(self.attention(x, attn_mask=attn_mask)))
        x = self.ff_norm(x + self.dropout(self.feed_forward(x)))
        return x

class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            BertEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        return x

class BertBase(nn.Module):
    def post_init(self) -> None:
        self._init_weights()

    def _init_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: self._init_module_weights(module, std=std))

    def _init_module_weights(self, module, std: float = 0.02):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

class BertPooler(nn.Module):
    """
    The pooler convert the output from the encoder of shape
    ``(batch_size, seq_length, hidden_size)`` to a tensor of shape
    ``(batch_size, hidden_size)`` (i.e. the representation corresponding to the first tokens).
    This is necessary for ``segment/sentence-level classification`` tasks.
    Ref: https://github.com/google-research/bert/blob/master/modeling.py#L219
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, encoder_output: Tensor) -> Tensor:
        sentence_representation = encoder_output[:, 0, :]
        pooled_output = self.dropout(sentence_representation)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return pooled_output

class Bert(BertBase):
    """
    The bare bert model without any task specific head on top of it.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.token_predicton_layer = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        encoder_output = self.embeddings(input_ids, segment_ids)
        encoder_output = self.encoder(encoder_output, mask=attn_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

class BertLMPredictionHead(nn.Module):
    """A head for masked language modeling task"""
    def __init__(self, config: BertConfig):
        super().__init__()

        # apply one more non-linear transformation before the output layer
        # ref: https://github.com/google-research/bert/blob/master/run_pretraining.py#L245
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            get_activation(config.activation),
            LayerNorm(config.hidden_size, config.norm_eps),
        )

        # for the decoder layer, we will use the same weights as the input embeddings
        # ref: https://github.com/google-research/bert/blob/master/run_pretraining.py#L257
        # ref: https://discuss.huggingface.co/t/understanding-bertlmpredictionhead/3618
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def set_output_weights(self, tokens_embeddings: nn.Embedding):
        assert self.decoder.weight.size() == tokens_embeddings.weight.size()
        self.decoder.weight = tokens_embeddings.weight

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform(x)
        x = self.decoder(x)
        return x

class BertNSPHead(nn.Module):
    """A head for next sentence prediction task"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.seq_relationship(x)
        return x

class BertForPretraining(BertBase):
    """Bert model with two heads on top of it

    Two heads including: a `masked language modeling` head and a `next sentence prediction` head
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = Bert(config)
        self.lm_head = BertLMPredictionHead(config)
        self.nsp_head = BertNSPHead(config)

        self.lm_head.set_output_weights(self.get_input_embeddings())
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.bert.embeddings.token_embeddings

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
        input_mask: Tensor | None = None,
        masked_positions: Tensor | None = None,
        masked_label_ids: Tensor | None = None,
        masked_weights: Tensor | None = None,
        next_sentence_labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """
        Args:
            input_ids: shape ``(batch_size, seq_length, hidden_size)``
            segment_ids: shape ``(batch_size, seq_length, hidden_size)``
            input_mask: shape ``(batch_size, seq_length, hidden_size)``
            masked_positions: shape ``(batch_size, max_masked_tokens)``
            masked_label_ids: shape ``(batch_size, max_masked_tokens)``
            masked_weights: shape ``(batch_size, max_masked_tokens)``
            next_sentence_labels: shape ``(batch_size, 1)``
        """

        # encoder_output: (batch_size, seq_length, hidden_size)
        # pooled_output: (batch_size, hidden_size)
        encoder_output, pooled_output = self.bert(
            input_ids,
            segment_ids=segment_ids,
            attn_mask=input_mask,
        )

        masked_lm_logits = self.lm_head(encoder_output)  # (batch_size, seq_length, vocab_size)
        nsp_logits = self.nsp_head(pooled_output)  # (batch_size, 2)
        masked_lm_loss, masked_lm_gathered_logits = self.get_masked_lm_loss(
            masked_lm_logits,
            masked_positions,
            masked_label_ids,
            masked_weights
        )
        nsp_loss = self.get_nsp_loss(nsp_logits, next_sentence_labels)

        return masked_lm_gathered_logits, nsp_logits, masked_lm_loss, nsp_loss

    def get_masked_lm_loss(
        self,
        masked_lm_logits: Tensor,
        masked_positions: Tensor | None,
        masked_label_ids: Tensor | None,
        masked_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor] | tuple[None, None]:
        if (
            masked_positions is None or
            masked_label_ids is None or
            masked_weights is None
        ):
            return None, None

        # in masked language modeling, we use negative log-likelihood,
        # all unmasked tokens are ignored when calculating the loss
        masked_lm_gathered_logits = gather_indices(masked_lm_logits, masked_positions)  # (batch_size * max_masked_tokens, hidden_size)

        log_probs = F.log_softmax(masked_lm_gathered_logits, dim=-1)  # (batch_size * max_masked_tokens, vocab_size)
        masked_label_ids = self._reshape_batched_tensor(masked_label_ids)  # (batch_size * max_masked_tokens)
        masked_weights = self._reshape_batched_tensor(masked_weights)  # (batch_size * max_masked_tokens)
        nll = -log_probs[range(len(masked_label_ids)), masked_label_ids] * masked_weights  # (batch_size * max_masked_tokens)
        masked_lm_loss = torch.sum(nll) / (torch.sum(masked_weights) + 1e-5)

        return masked_lm_loss, masked_lm_gathered_logits

    def get_nsp_loss(
        self,
        nsp_logits: Tensor,
        next_sentence_label: Tensor | None
    ) -> Tensor | None:
        if next_sentence_label is None:
            return None

        # in next sentence prediction, we also use log-likelihood
        log_probs = F.log_softmax(nsp_logits, dim=-1)  # (batch_size, 2)
        nll = -log_probs[range(len(next_sentence_label)), next_sentence_label]  # (batch_size,)
        nsp_loss = torch.mean(nll)
        return nsp_loss

    def _reshape_batched_tensor(self, x: Tensor) -> Tensor:
        assert 1 <= x.dim() and x.dim() <= 3
        if x.dim() == 2:
            return x.view(-1)  # (batch_size * seq_length)
        return x.view(-1, x.size(-1))  # (batch_size * seq_length, ...)

def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout: float | nn.Dropout | None = None,
) -> Tensor:
    """
    Args:
        query (Tensor): query tensor, shape ``(batch_size, num_heads, q_length, d_k)``
        key (Tensor): key tensor, shape ``(batch_size, num_heads, k_length, d_k)``
        value (Tensor): value tensor, shape ``(batch_size, num_heads, k_length, d_v)``
        attn_mask (Tensor): mask tensor, shape ``(batch_size, 1, 1, k_length)`` or ``(batch_size, k_length)`` (default: None)
        dropout (nn.Dropout): dropout layer (default: None)

    Returns:
        Tensor, shape ``(batch_size, num_heads, q_length, d_v)``
    """
    if attn_mask is not None:
        assert attn_mask.dim() == 2 or attn_mask.dim() == 4
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

    d_k = query.size(-1)
    attention_probs = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        attention_probs.masked_fill_(attn_mask == False, float('-inf'))

    attention_probs = F.softmax(attention_probs, dim=-1)
    if dropout is not None:
        if isinstance(dropout, float):
            dropout = nn.Dropout(dropout)
        attention_probs = dropout(attention_probs)

    output = attention_probs @ value
    return output

def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(
            f'Unrecognized activation type: {act_type}. '
            'Possible values are "relu", "gelu".'
        )

def get_device(device: torch.device | str = 'auto') -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def gather_indices(x: Tensor, positions: Tensor) -> Tensor:
    """
    Args:
        x (Tensor): shape ``(batch_size, seq_length, hidden_size``
        positions (Tensor): shape ``(batch_size, max_masked_tokens)``

    Returns:
        gathered Tensor: shape ``(batch_size, max_masked_tokens, hidden_size)``
    """

    batch_size, seq_length, _ = x.size()
    flat_offset = torch.arange(0, batch_size) * seq_length
    flat_offset = flat_offset.unsqueeze(1).to(positions.device)

    flat_positions = positions + flat_offset
    flat_positions = flat_positions.view(-1)  # (batch_size * max_masked_tokens)
    x = x.view(batch_size * seq_length, -1)  # (batch_size * seq_length, hidden_size)
    return x[flat_positions]
