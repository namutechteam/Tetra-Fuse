# Copyright (c) DP Technology. MIT License.
# Vendored standalone version of the Uni-Mol2 model forward pass,
# extracted from:
#   https://github.com/deepmodeling/Uni-Mol
#     unimol_tools/unimol_tools/models/transformersv2.py
#     unimol_tools/unimol_tools/models/unimolv2.py
#     unimol_tools/unimol_tools/data/conformer.py   (featurizer)
#     unimol_tools/unimol_tools/utils/util.py       (padding helpers)
#
# Dependencies are limited to: torch, numpy, math, functools (stdlib).
# RDKit is only required if you use `mol2unimolv2` / `featurize_mols`.
# No unicore / fairseq / addict / numba dependencies are required at import time.

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Utility layers (previously imported from unicore or defined in unimol_tools)
# ---------------------------------------------------------------------------


def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None, inplace=True):
    """Softmax (with optional additive mask/bias) followed by dropout.

    Replaces `unicore.modules.softmax_dropout`.
    """
    input = input.contiguous()
    if not inplace:
        input = input.clone()
    if mask is not None:
        input = input + mask
    if bias is not None:
        input = input + bias
    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)


def permute_final_dims(tensor: torch.Tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and self.training:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        return x


class Linear(nn.Linear):
    """Linear layer with configurable init (replaces unicore Linear)."""

    def __init__(self, d_in: int, d_out: int, bias: bool = True, init: str = "default"):
        super().__init__(d_in, d_out, bias=bias)
        self.use_bias = bias
        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            nn.init.xavier_uniform_(self.weight, gain=1)
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError(f"Invalid init method: {init}")

    def _trunc_normal_init(self, scale=1.0):
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale ** 0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                self.bias.fill_(1.0)


class Embedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Transformer building blocks  (from transformersv2.py)
# ---------------------------------------------------------------------------


class Transition(nn.Module):
    def __init__(self, d_in, n, dropout=0.0):
        super().__init__()
        self.linear_1 = Linear(d_in, n * d_in, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(n * d_in, d_in, init="final")
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)
        return x


class Attention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, pair_dim, head_dim, num_heads,
                 gating: bool = False, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        total_dim = head_dim * num_heads
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")
        self.norm = head_dim ** -0.5
        self.dropout = dropout
        self.linear_bias = Linear(pair_dim, num_heads)

    def forward(self, q, k, v, pair, mask: Optional[torch.Tensor] = None):
        g = None
        if self.linear_g is not None:
            g = self.linear_g(q)
        q = self.linear_q(q)
        q = q * self.norm
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)

        attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        attn = softmax_dropout(attn, self.dropout, self.training, mask=mask, bias=bias)
        o = torch.matmul(attn, v)
        del attn, v

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)
        if g is not None:
            o = torch.sigmoid(g) * o
        o = self.linear_o(o)
        return o


class OuterProduct(nn.Module):
    def __init__(self, d_atom, d_pair, d_hid=32):
        super().__init__()
        self.linear_in = nn.Linear(d_atom, d_hid * 2)
        self.linear_out = nn.Linear(d_hid ** 2, d_pair)
        self.act = nn.GELU()
        self._memory_efficient = True

    def _opm(self, a, b):
        bsz, n, d = a.shape
        a = a.view(bsz, n, 1, d, 1)
        b = b.view(bsz, 1, n, 1, d)
        outer = a * b
        outer = outer.view(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    def forward(self, m, op_mask=None, op_norm=None):
        ab = self.linear_in(m)
        ab = ab * op_mask
        a, b = ab.chunk(2, dim=-1)
        if self._memory_efficient and torch.is_grad_enabled():
            z = checkpoint(self._opm, a, b, use_reentrant=False)
        else:
            z = self._opm(a, b)
        z = z * op_norm
        return z


class AtomFeature(nn.Module):
    """Initial per-atom embedding from atom_feat + degree + token embeddings."""

    def __init__(self, num_atom, num_degree, hidden_dim):
        super().__init__()
        self.atom_encoder = Embedding(num_atom, hidden_dim, padding_idx=0)
        self.degree_encoder = Embedding(num_degree, hidden_dim, padding_idx=0)
        self.vnode_encoder = Embedding(1, hidden_dim)

    def forward(self, batched_data, token_feat):
        x = batched_data["atom_feat"]
        degree = batched_data["degree"]
        n_graph, n_node = x.size()[:2]
        node_feature = self.atom_encoder(x).sum(dim=-2)
        dtype = node_feature.dtype
        degree_feature = self.degree_encoder(degree)
        node_feature = node_feature + degree_feature + token_feat
        graph_token_feature = self.vnode_encoder.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature.type(dtype)


class EdgeFeature(nn.Module):
    """Attention bias from shortest-path + edge embeddings."""

    def __init__(self, pair_dim, num_edge, num_spatial):
        super().__init__()
        self.pair_dim = pair_dim
        self.edge_encoder = Embedding(num_edge, pair_dim, padding_idx=0)
        self.shorest_path_encoder = Embedding(num_spatial, pair_dim, padding_idx=0)
        self.vnode_virtual_distance = Embedding(1, pair_dim)

    def forward(self, batched_data, graph_attn_bias):
        shortest_path = batched_data["shortest_path"]
        edge_input = batched_data["edge_feat"]
        graph_attn_bias[:, 1:, 1:, :] = self.shorest_path_encoder(shortest_path)
        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, 1:, 0, :] = t
        graph_attn_bias[:, 0, :, :] = t
        edge_input = self.edge_encoder(edge_input).mean(-2)
        graph_attn_bias[:, 1:, 1:, :] = graph_attn_bias[:, 1:, 1:, :] + edge_input
        return graph_attn_bias


@torch.jit.script
def _gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianKernel(nn.Module):
    def __init__(self, K=128, num_pair=512, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.K = K
        mean = torch.linspace(start, stop, K)
        self.std = std_width * (mean[1] - mean[0])
        self.register_buffer("mean", mean)
        self.mul = Embedding(num_pair, 1, padding_idx=0)
        self.bias = Embedding(num_pair, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x, atom_pair):
        mul = self.mul(atom_pair).abs().sum(dim=-2)
        bias = self.bias(atom_pair).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.mean.float().view(-1)
        return _gaussian(x.float(), mean, self.std)


class NonLinear(nn.Module):
    def __init__(self, input_dim, output_size, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = input_dim
        self.layer1 = Linear(input_dim, hidden, init="relu")
        self.layer2 = Linear(hidden, output_size, init="final")

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class SE3InvariantKernel(nn.Module):
    def __init__(self, pair_dim, num_pair, num_kernel, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.num_kernel = num_kernel
        self.gaussian = GaussianKernel(num_kernel, num_pair,
                                       std_width=std_width, start=start, stop=stop)
        self.out_proj = NonLinear(num_kernel, pair_dim)

    def forward(self, dist, node_type_edge):
        edge_feature = self.gaussian(dist, node_type_edge.long())
        edge_feature = self.out_proj(edge_feature)
        return edge_feature


class MovementPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, pair_dim: int, num_head: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.q_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.num_head = num_head
        self.scaling = (embed_dim // num_head) ** -0.5
        self.force_proj1 = Linear(embed_dim, 1, init="final", bias=False)
        self.linear_bias = Linear(pair_dim, num_head)
        self.pair_layer_norm = nn.LayerNorm(pair_dim)
        self.dropout = 0.1

    def forward(self, query, pair, attn_mask, delta_pos):
        bsz, n_node, _ = query.size()
        query = self.layer_norm(query)
        q = (self.q_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
             * self.scaling)
        k = self.k_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)

        pair = self.pair_layer_norm(pair)
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        attn_probs = softmax_dropout(
            attn, self.dropout, self.training,
            mask=attn_mask.contiguous(), bias=bias.contiguous(),
        ).view(bsz, self.num_head, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(attn_probs)
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        cur_force = self.force_proj1(x).view(bsz, n_node, 3)
        return cur_force


class DropPath(nn.Module):
    def __init__(self, prob=None):
        super().__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid):
        super().__init__()
        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init="gating")
        self.linear_g = Linear(d_pair, d_pair, init="gating")
        self.linear_z = Linear(d_hid, d_pair, init="final")
        self.layer_norm_out = nn.LayerNorm(d_hid)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        mask = mask * (mask.shape[-2] ** -0.5)
        g = self.linear_g(z)
        if self.training:
            ab = self.linear_ab_p(z) * mask * torch.sigmoid(self.linear_ab_g(z))
        else:
            ab = self.linear_ab_p(z)
            ab = ab * mask
            ab = ab * torch.sigmoid(self.linear_ab_g(z))
        a, b = torch.chunk(ab, 2, dim=-1)
        del z, ab
        a1 = permute_final_dims(a, (2, 0, 1))
        b1 = b.transpose(-1, -3)
        x = torch.matmul(a1, b1)
        del a1, b1
        b2 = permute_final_dims(b, (2, 0, 1))
        a2 = a.transpose(-1, -3)
        x = x + torch.matmul(a2, b2)
        del a, b, a2, b2
        x = permute_final_dims(x, (1, 2, 0))
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        return g * x


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "tanh":
        return torch.tanh
    if activation == "linear":
        return lambda x: x
    raise RuntimeError(f"--activation-fn {activation} not supported")


class TransformerEncoderLayerV2(nn.Module):
    def __init__(self, embedding_dim=768, pair_dim=64, pair_hidden_dim=32,
                 ffn_embedding_dim=3072, num_attention_heads=8,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.1,
                 activation_fn="gelu", droppath_prob=0.0, pair_dropout=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = Dropout(dropout)

        self.activation_fn = _get_activation_fn(activation_fn)
        head_dim = embedding_dim // num_attention_heads
        self.self_attn = Attention(
            embedding_dim, embedding_dim, embedding_dim,
            pair_dim=pair_dim, head_dim=head_dim,
            num_heads=num_attention_heads, gating=False,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.ffn = Transition(embedding_dim, ffn_embedding_dim // embedding_dim,
                              dropout=activation_dropout)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.x_layer_norm_opm = nn.LayerNorm(embedding_dim)

        self.opm = OuterProduct(embedding_dim, pair_dim, d_hid=pair_hidden_dim)
        self.pair_layer_norm_ffn = nn.LayerNorm(pair_dim)
        self.pair_ffn = Transition(pair_dim, 1, dropout=activation_dropout)

        self.pair_dropout = pair_dropout
        self.pair_layer_norm_trimul = nn.LayerNorm(pair_dim)
        self.pair_tri_mul = TriangleMultiplication(pair_dim, pair_hidden_dim)

    def shared_dropout(self, x, shared_dim, dropout):
        shape = list(x.shape)
        shape[shared_dim] = 1
        with torch.no_grad():
            mask = x.new_ones(shape)
        return F.dropout(mask, p=dropout, training=self.training) * x

    def forward(self, x, pair, pair_mask, self_attn_mask=None, op_mask=None, op_norm=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, pair=pair, mask=self_attn_mask)
        x = self.dropout_module(x)
        x = residual + x
        x = x + self.dropout_module(self.ffn(self.final_layer_norm(x)))
        pair = pair + self.dropout_module(self.opm(self.x_layer_norm_opm(x), op_mask, op_norm))
        pair = pair + self.shared_dropout(
            self.pair_tri_mul(self.pair_layer_norm_trimul(pair), pair_mask),
            -3, self.pair_dropout,
        )
        pair = pair + self.dropout_module(self.pair_ffn(self.pair_layer_norm_ffn(pair)))
        return x, pair


class TransformerEncoderWithPairV2(nn.Module):
    def __init__(self, num_encoder_layers=6, embedding_dim=768, pair_dim=64,
                 pair_hidden_dim=32, ffn_embedding_dim=3072, num_attention_heads=8,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.0,
                 activation_fn="gelu", droppath_prob=0.0, pair_dropout=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_attention_heads
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.pair_layer_norm = nn.LayerNorm(pair_dim)
        self.layers = nn.ModuleList()

        if droppath_prob > 0:
            droppath_probs = [x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)]
        else:
            droppath_probs = None

        self.layers.extend([
            TransformerEncoderLayerV2(
                embedding_dim=embedding_dim, pair_dim=pair_dim,
                pair_hidden_dim=pair_hidden_dim, ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout, attention_dropout=attention_dropout,
                activation_dropout=activation_dropout, activation_fn=activation_fn,
                droppath_prob=droppath_probs[i] if droppath_probs is not None else 0,
                pair_dropout=pair_dropout,
            )
            for i in range(num_encoder_layers)
        ])

    def forward(self, x, pair, atom_mask, pair_mask, attn_mask=None):
        x = self.layer_norm(x)
        pair = self.pair_layer_norm(pair)
        op_mask = atom_mask.unsqueeze(-1)
        op_mask = op_mask * (op_mask.size(-2) ** -0.5)
        eps = 1e-3
        op_norm = 1.0 / (eps + torch.einsum("...bc,...dc->...bdc", op_mask, op_mask))
        for layer in self.layers:
            x, pair = layer(x, pair, pair_mask=pair_mask,
                            self_attn_mask=attn_mask, op_mask=op_mask, op_norm=op_norm)
        return x, pair


# ---------------------------------------------------------------------------
# Config presets  (from molecule_architecture in unimolv2.py)
# ---------------------------------------------------------------------------


def _molecule_architecture(model_size: str = "84m") -> Dict[str, object]:
    args: Dict[str, object] = {}
    if model_size == "84m":
        args.update(num_encoder_layers=12, encoder_embed_dim=768,
                    num_attention_heads=48, ffn_embedding_dim=768,
                    encoder_attention_heads=48)
    elif model_size == "164m":
        args.update(num_encoder_layers=24, encoder_embed_dim=768,
                    num_attention_heads=48, ffn_embedding_dim=768,
                    encoder_attention_heads=48)
    elif model_size == "310m":
        args.update(num_encoder_layers=32, encoder_embed_dim=1024,
                    num_attention_heads=64, ffn_embedding_dim=1024,
                    encoder_attention_heads=64)
    elif model_size == "570m":
        args.update(num_encoder_layers=32, encoder_embed_dim=1536,
                    num_attention_heads=96, ffn_embedding_dim=1536,
                    encoder_attention_heads=96)
    elif model_size == "1.1B":
        args.update(num_encoder_layers=64, encoder_embed_dim=1536,
                    num_attention_heads=96, ffn_embedding_dim=1536,
                    encoder_attention_heads=96)
    else:
        raise ValueError(f"Unknown model_size: {model_size}")
    args.update(
        pair_embed_dim=512, pair_hidden_dim=64,
        dropout=0.1, attention_dropout=0.1, activation_dropout=0.0,
        activation_fn="gelu", droppath_prob=0.0, pair_dropout=0.25,
        backbone="transformer",
        gaussian_std_width=1.0, gaussian_mean_start=0.0, gaussian_mean_stop=9.0,
        pooler_dropout=0.0, pooler_activation_fn="tanh",
    )
    return args


# ---------------------------------------------------------------------------
# UniMolV2 model (feature extractor variant, no classification head by default)
# ---------------------------------------------------------------------------


class UniMolV2Model(nn.Module):
    """UniMol2 backbone. Forward produces per-atom and CLS representations.

    Unlike the upstream model, this version does NOT include a classification
    head and does NOT auto-download weights. Use `load_unimol2_weights` to load
    a pretrained .pt checkpoint.
    """

    def __init__(self, model_size: str = "84m"):
        super().__init__()
        args = _molecule_architecture(model_size)
        self.args = args
        self.model_size = model_size

        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = 127
        self.embed_tokens = nn.Embedding(self.token_num, args["encoder_embed_dim"], self.padding_idx)

        self.encoder = TransformerEncoderWithPairV2(
            num_encoder_layers=args["num_encoder_layers"],
            embedding_dim=args["encoder_embed_dim"],
            pair_dim=args["pair_embed_dim"],
            pair_hidden_dim=args["pair_hidden_dim"],
            ffn_embedding_dim=args["ffn_embedding_dim"],
            num_attention_heads=args["num_attention_heads"],
            dropout=args["dropout"],
            attention_dropout=args["attention_dropout"],
            activation_dropout=args["activation_dropout"],
            activation_fn=args["activation_fn"],
            droppath_prob=args["droppath_prob"],
            pair_dropout=args["pair_dropout"],
        )

        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512
        K = 128

        self.atom_feature = AtomFeature(num_atom=num_atom, num_degree=num_degree,
                                        hidden_dim=args["encoder_embed_dim"])
        self.edge_feature = EdgeFeature(pair_dim=args["pair_embed_dim"],
                                        num_edge=num_edge, num_spatial=num_spatial)
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=args["pair_embed_dim"], num_pair=num_pair, num_kernel=K,
            std_width=args["gaussian_std_width"],
            start=args["gaussian_mean_start"], stop=args["gaussian_mean_stop"],
        )
        self.movement_pred_head = MovementPredictionHead(
            args["encoder_embed_dim"], args["pair_embed_dim"],
            args["encoder_attention_heads"],
        )
        self.dtype = torch.float32

    @property
    def embed_dim(self) -> int:
        return int(self.args["encoder_embed_dim"])

    def forward(self, atom_feat, atom_mask, edge_feat, shortest_path, degree,
                pair_type, attn_bias, src_tokens, src_coord,
                return_atomic_reprs: bool = False) -> Dict[str, torch.Tensor]:
        """Returns dict with keys: 'cls_repr' [B, D], 'all_repr' [B, N+1, D],
        and if return_atomic_reprs=True also 'atomic_reprs' (list), etc."""
        pos = src_coord
        n_mol, n_atom = atom_feat.shape[:2]
        token_feat = self.embed_tokens(src_tokens)
        x = self.atom_feature({"atom_feat": atom_feat, "degree": degree}, token_feat)
        dtype = self.dtype
        x = x.type(dtype)

        attn_mask = attn_bias.clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.args["encoder_attention_heads"], 1, 1
        )
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args["pair_embed_dim"])
        attn_bias = self.edge_feature(
            {"shortest_path": shortest_path, "edge_feat": edge_feat}, attn_bias
        )
        attn_mask = attn_mask.type(self.dtype)

        atom_mask_cls = torch.cat([
            torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype),
            atom_mask,
        ], dim=1).type(self.dtype)
        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
        new_attn_bias = attn_bias.clone()
        new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d
        new_attn_bias = new_attn_bias.type(dtype)

        x, pair = self.encoder(
            x, new_attn_bias, atom_mask=atom_mask_cls,
            pair_mask=pair_mask, attn_mask=attn_mask,
        )
        # NOTE: movement_pred_head output is discarded here (we only need reprs).
        # Keeping the module so state_dict keys match the published checkpoint.

        cls_repr = x[:, 0, :]
        out: Dict[str, torch.Tensor] = {"cls_repr": cls_repr, "all_repr": x}

        if return_atomic_reprs:
            filtered_tensors: List[torch.Tensor] = []
            filtered_coords: List[torch.Tensor] = []
            for tokens, coord in zip(src_tokens, src_coord):
                m = (tokens != 0) & (tokens != 1) & (tokens != 2)
                filtered_tensors.append(tokens[m])
                filtered_coords.append(coord[m])
            lengths = [len(t) for t in filtered_tensors]
            cls_atomic_reprs: List[torch.Tensor] = []
            for i in range(len(filtered_tensors)):
                cls_atomic_reprs.append(x[i, 1: lengths[i] + 1, :])
            out.update({
                "atomic_symbol": filtered_tensors,
                "atomic_coords": filtered_coords,
                "atomic_reprs": cls_atomic_reprs,
            })
        return out


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------


def load_unimol2_weights(model: nn.Module, ckpt_path: str, strict: bool = False):
    """Load a Uni-Mol2 .pt checkpoint into `model`.

    Accepts checkpoints that wrap the state dict under 'model' or
    'model_state_dict' keys, as produced by the upstream repo /
    HuggingFace `dptech/Uni-Mol2` .pt files.

    Keys not present in the feature-extractor model (e.g. `classification_head.*`)
    are simply ignored when strict=False.
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict):
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], dict):
            state_dict = state_dict["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}


# ---------------------------------------------------------------------------
# Featurizer: (rdkit Mol) -> input dict  (mirrors mol2unimolv2)
# ---------------------------------------------------------------------------

# These lists match unimol_tools/data/conformer.py exactly so that the
# pretrained embeddings line up with the integer indices we produce.

ALLOWABLE_FEATURES = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL", "CHI_OCTAHEDRAL", "CHI_SQUAREPLANAR", "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def _safe_index(lst, e):
    try:
        return lst.index(e)
    except ValueError:
        return len(lst) - 1


def _atom_to_feature_vector(atom) -> List[int]:
    af = ALLOWABLE_FEATURES
    return [
        _safe_index(af["possible_atomic_num_list"], atom.GetAtomicNum()),
        af["possible_chirality_list"].index(str(atom.GetChiralTag())),
        _safe_index(af["possible_degree_list"], atom.GetTotalDegree()),
        _safe_index(af["possible_formal_charge_list"], atom.GetFormalCharge()),
        _safe_index(af["possible_numH_list"], atom.GetTotalNumHs()),
        _safe_index(af["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()),
        _safe_index(af["possible_hybridization_list"], str(atom.GetHybridization())),
        af["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        af["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]


def _bond_to_feature_vector(bond) -> List[int]:
    af = ALLOWABLE_FEATURES
    return [
        _safe_index(af["possible_bond_type_list"], str(bond.GetBondType())),
        af["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        af["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]


def _convert_to_single_emb(x: np.ndarray, sizes: List[int]) -> np.ndarray:
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


def _floyd_warshall(M: np.ndarray) -> np.ndarray:
    """Pure-numpy floyd-warshall. Upstream uses numba @njit; we avoid that dep."""
    nrows, ncols = M.shape
    assert nrows == ncols
    n = nrows
    D = M.astype(np.int64, copy=True)
    # unreachable -> 510
    D[D == 0] = 510
    for i in range(n):
        D[i, i] = 0
    for k in range(n):
        # Vectorized relaxation: D = min(D, D[:,k:k+1] + D[k:k+1,:])
        via_k = D[:, k:k + 1] + D[k:k + 1, :]
        np.minimum(D, via_k, out=D)
    D[D >= 510] = 510
    return D


def mol2unimolv2(mol, max_atoms: int = 128, remove_hs: bool = True) -> Dict[str, np.ndarray]:
    """Convert an RDKit Mol (with a 3D conformer) to Uni-Mol2 input arrays.

    Requires rdkit at call time (not at import time).
    """
    from rdkit.Chem import AllChem  # local import

    mol = AllChem.RemoveAllHs(mol)
    atoms = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    coords = mol.GetConformer().GetPositions().astype(np.float32)

    if len(atoms) > max_atoms:
        mask = np.zeros(len(atoms), dtype=bool)
        mask[:max_atoms] = True
        np.random.shuffle(mask)
        atoms = atoms[mask]
        coords = coords[mask]
    else:
        mask = np.ones(len(atoms), dtype=bool)

    src_tokens = np.array(
        [AllChem.GetPeriodicTable().GetAtomicNumber(sym) for sym in atoms],
        dtype=np.int64,
    )
    src_coord = coords

    # graph features
    node_attr = np.array([_atom_to_feature_vector(a) for a in mol.GetAtoms()], dtype=np.int32)
    if len(mol.GetBonds()) > 0:
        edges, eattrs = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            fb = _bond_to_feature_vector(b)
            edges.append((i, j)); eattrs.append(fb)
            edges.append((j, i)); eattrs.append(fb)
        edge_index = np.array(edges, dtype=np.int32).T
        edge_attr = np.array(eattrs, dtype=np.int32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, 3), dtype=np.int32)

    feat = _get_graph_features(edge_attr, edge_index, node_attr, drop_feat=0, mask=mask)
    feat["src_tokens"] = src_tokens
    feat["src_coord"] = src_coord
    return feat


def _get_graph_features(edge_attr, edge_index, node_attr, drop_feat, mask):
    atom_feat_sizes = [16 for _ in range(8)]
    edge_feat_sizes = [16, 16, 16]
    x = node_attr
    N = x.shape[0]

    atom_feat = _convert_to_single_emb(x[:, 1:].copy(), atom_feat_sizes)

    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    edge_feat = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    edge_feat[edge_index[0, :], edge_index[1, :]] = (
        _convert_to_single_emb(edge_attr.copy(), edge_feat_sizes) + 1
    )

    shortest_path_result = _floyd_warshall(adj.copy())

    if drop_feat:
        atom_feat[...] = 1
        edge_feat[...] = 1
        degree[...] = 1
        shortest_path_result[...] = 511
    else:
        atom_feat = atom_feat + 2
        edge_feat = edge_feat + 2
        degree = degree + 2
        shortest_path_result = shortest_path_result + 1

    feat = {}
    feat["atom_feat"] = atom_feat[mask]
    feat["atom_mask"] = np.ones(N, dtype=np.int64)[mask]
    feat["edge_feat"] = edge_feat[mask][:, mask]
    feat["shortest_path"] = shortest_path_result[mask][:, mask]
    feat["degree"] = degree.reshape(-1)[mask]

    atoms_ids = atom_feat[..., 0]
    pair_type = np.concatenate([
        np.expand_dims(atoms_ids, axis=(1, 2)).repeat(N, axis=1),
        np.expand_dims(atoms_ids, axis=(0, 2)).repeat(N, axis=0),
    ], axis=-1)
    pair_type = pair_type[mask][:, mask]
    feat["pair_type"] = _convert_to_single_emb(pair_type, [128, 128])

    n_sel = int(mask.sum())
    feat["attn_bias"] = np.zeros((n_sel + 1, n_sel + 1), dtype=np.float32)
    return feat


# ---------------------------------------------------------------------------
# Collate helpers (replace pad_1d_tokens / pad_2d / pad_coords from utils)
# ---------------------------------------------------------------------------


def _pad_1d(values: List[torch.Tensor], pad_idx=0) -> torch.Tensor:
    size = max(v.size(0) for v in values)
    res = values[0].new_full((len(values), size), pad_idx)
    for i, v in enumerate(values):
        res[i, : len(v)] = v
    return res


def _pad_2d(values: List[torch.Tensor], pad_idx=0, dim: int = 1) -> torch.Tensor:
    size = max(v.size(0) for v in values)
    if dim == 1:
        res = values[0].new_full((len(values), size, size), pad_idx)
    else:
        res = values[0].new_full((len(values), size, size, dim), pad_idx)
    for i, v in enumerate(values):
        res[i, : len(v), : len(v)] = v
    return res


def _pad_coords(values: List[torch.Tensor], pad_idx=0, dim: int = 3) -> torch.Tensor:
    size = max(v.size(0) for v in values)
    res = values[0].new_full((len(values), size, dim), pad_idx)
    for i, v in enumerate(values):
        res[i, : len(v), :] = v
    return res


def collate_unimolv2(samples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Collate a list of featurizer outputs into a batched tensor dict ready
    for `UniMolV2Model.forward`. pad_idx is 0."""
    batch: Dict[str, torch.Tensor] = {}
    keys = samples[0].keys()
    for k in keys:
        ts = [torch.as_tensor(s[k]) for s in samples]
        if k == "atom_feat":
            batch[k] = _pad_coords(ts, pad_idx=0, dim=8)
        elif k == "atom_mask":
            batch[k] = _pad_1d(ts, pad_idx=0)
        elif k == "edge_feat":
            batch[k] = _pad_2d(ts, pad_idx=0, dim=3)
        elif k == "shortest_path":
            batch[k] = _pad_2d(ts, pad_idx=0)
        elif k == "degree":
            batch[k] = _pad_1d(ts, pad_idx=0)
        elif k == "pair_type":
            batch[k] = _pad_2d(ts, pad_idx=0, dim=2)
        elif k == "attn_bias":
            batch[k] = _pad_2d(ts, pad_idx=0)
        elif k == "src_tokens":
            batch[k] = _pad_1d(ts, pad_idx=0)
        elif k == "src_coord":
            batch[k] = _pad_coords(ts, pad_idx=0)
        else:
            # fallback: stack
            batch[k] = torch.stack(ts, dim=0)
    # Ensure integer tensors have integer dtype for Embedding lookups
    for k in ("atom_feat", "edge_feat", "shortest_path", "degree",
              "pair_type", "src_tokens", "atom_mask"):
        if k in batch:
            batch[k] = batch[k].long()
    if "src_coord" in batch:
        batch["src_coord"] = batch["src_coord"].float()
    if "attn_bias" in batch:
        batch["attn_bias"] = batch["attn_bias"].float()
    return batch


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _make_random_batch(n_mol: int = 2, n_atom: int = 40, device: str = "cpu"):
    """Build a dummy batch directly (no RDKit needed) for smoke tests."""
    torch.manual_seed(0)
    np.random.seed(0)

    # atom_feat: [B, N, 8] long, values in [2, 2+16) per-slot after offset scheme
    atom_feat = torch.randint(2, 2 + 16, (n_mol, n_atom, 8), dtype=torch.long)
    # atom_mask: [B, N] -> 1 for all atoms
    atom_mask = torch.ones(n_mol, n_atom, dtype=torch.long)
    # edge_feat: [B, N, N, 3], values in [2, 2+16)
    edge_feat = torch.randint(2, 2 + 16, (n_mol, n_atom, n_atom, 3), dtype=torch.long)
    # shortest_path: [B, N, N], values in [1, 512)
    shortest_path = torch.randint(1, 20, (n_mol, n_atom, n_atom), dtype=torch.long)
    # degree: [B, N], values in [2, 2+16)
    degree = torch.randint(2, 2 + 16, (n_mol, n_atom), dtype=torch.long)
    # pair_type: [B, N, N, 2] long
    pair_type = torch.randint(1, 128, (n_mol, n_atom, n_atom, 2), dtype=torch.long)
    # attn_bias: [B, N+1, N+1] float
    attn_bias = torch.zeros(n_mol, n_atom + 1, n_atom + 1, dtype=torch.float32)
    # src_tokens: [B, N] in [3, token_num), with 0/1/2 reserved
    src_tokens = torch.randint(3, 128, (n_mol, n_atom), dtype=torch.long)
    # src_coord: [B, N, 3] float
    src_coord = torch.randn(n_mol, n_atom, 3, dtype=torch.float32)

    batch = dict(
        atom_feat=atom_feat, atom_mask=atom_mask, edge_feat=edge_feat,
        shortest_path=shortest_path, degree=degree, pair_type=pair_type,
        attn_bias=attn_bias, src_tokens=src_tokens, src_coord=src_coord,
    )
    return {k: v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[unimol_v2 smoke] device={device}")

    model = UniMolV2Model(model_size="84m").to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[unimol_v2 smoke] 84m param count: {n_params/1e6:.1f}M")

    batch = _make_random_batch(n_mol=2, n_atom=40, device=device)
    t0 = time.time()
    with torch.no_grad():
        out = model(**batch)
    dt = time.time() - t0

    print(f"[unimol_v2 smoke] forward OK in {dt:.2f}s")
    print(f"[unimol_v2 smoke] cls_repr shape: {tuple(out['cls_repr'].shape)}")
    print(f"[unimol_v2 smoke] all_repr shape: {tuple(out['all_repr'].shape)}")
    assert out["cls_repr"].shape == (2, model.embed_dim), "unexpected cls shape"
    print("[unimol_v2 smoke] PASS")
