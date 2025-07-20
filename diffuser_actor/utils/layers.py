import numpy as np
from torch import nn
from torch.nn import functional as F

from mamba_ssm import Mamba


class ParallelAttentionLayer(nn.Module):
    """Self-/Cross-attention between two sequences using Mamba."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8, pre_norm=False,
                 self_attention1=True, self_attention2=True,
                 cross_attention1=True, cross_attention2=True,
                 apply_ffn=True,
                 slot_attention12=False, slot_attention21=False,
                 rotary_pe=False, use_adaln=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attention1 = self_attention1
        self.self_attention2 = self_attention2
        self.cross_attention1 = cross_attention1
        self.cross_attention2 = cross_attention2
        self.apply_ffn = apply_ffn
        self.rotary_pe = rotary_pe

        # Self-attention for seq1
        if self.self_attention1:
            self.adaln_1 = None
            if use_adaln:
                self.adaln_1 = AdaLN(d_model)
            self.sa1 = Mamba(d_model=d_model)  # Mamba with headdim
            self.dropout_1 = nn.Dropout(dropout)
            self.norm_1 = nn.LayerNorm(d_model)

        # Self-attention for seq2
        if self.self_attention2:
            self.adaln_2 = None
            if use_adaln:
                self.adaln_2 = AdaLN(d_model)
            self.sa2 = Mamba(d_model=d_model)  # Mamba with headdim
            self.dropout_2 = nn.Dropout(dropout)
            self.norm_2 = nn.LayerNorm(d_model)

        # Cross attention from seq1 to seq2
        self.norm_12 = None
        if cross_attention1:
            self.adaln_12 = None
            if use_adaln:
                self.adaln_12 = AdaLN(d_model)
            self.cross_12 = Mamba(d_model=d_model)  # Mamba with headdim
            self.dropout_12 = nn.Dropout(dropout)
            self.norm_12 = nn.LayerNorm(d_model)

        # Cross attention from seq2 to seq1
        self.norm_21 = None
        if cross_attention2:
            self.adaln_21 = None
            if use_adaln:
                self.adaln_21 = AdaLN(d_model)
            self.cross_21 = Mamba(d_model=d_model)  # Mamba with headdim
            self.dropout_21 = nn.Dropout(dropout)
            self.norm_21 = nn.LayerNorm(d_model)

        # FFN-1
        if self_attention1 or cross_attention1:
            self.adaln_ff1 = None
            if use_adaln:
                self.adaln_ff1 = AdaLN(d_model)
            self.ffn_12 = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout)
            )
            self.norm_122 = nn.LayerNorm(d_model)

        # FFN-2
        if self_attention2 or cross_attention2:
            self.adaln_ff2 = None
            if use_adaln:
                self.adaln_ff2 = AdaLN(d_model)
            self.ffn_21 = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout)
            )
            self.norm_212 = nn.LayerNorm(d_model)

    def _norm(self, x, layer, normalize=True):
        if normalize and layer is not None:
            return layer(x)
        return x

    def forward(self, seq1, seq1_key_padding_mask, seq2,
                seq2_key_padding_mask,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                ada_sgnl=None):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        
        # Cross-attention from seq1 to seq2 using Mamba
        if self.cross_attention1:
            seq1b = self.cross_12(seq1, inference_params=None)
            seq1 = seq1 + self.dropout_12(seq1b)
            seq1 = self._norm(seq1, self.norm_12, not self.pre_norm)

        # Cross-attention from seq2 to seq1 using Mamba
        if self.cross_attention2:
            seq2b = self.cross_21(seq2, inference_params=None)
            seq2 = seq2 + self.dropout_21(seq2b)
            seq2 = self._norm(seq2, self.norm_21, not self.pre_norm)

        # Self-attention for seq1 using Mamba
        if self.self_attention1:
            seq1b = self.sa1(seq1, inference_params=None)
            seq1 = seq1 + self.dropout_1(seq1b)
            seq1 = self._norm(seq1, self.norm_1, not self.pre_norm)

        # Self-attention for seq2 using Mamba
        if self.self_attention2:
            seq2b = self.sa2(seq2, inference_params=None)
            seq2 = seq2 + self.dropout_2(seq2b)
            seq2 = self._norm(seq2, self.norm_2, not self.pre_norm)

        return seq1, seq2


class ParallelAttention(nn.Module):
    """Self-/Cross-attention between two sequences using Mamba."""

    def __init__(self, num_layers=1,
                 d_model=256, dropout=0.1, n_heads=8, pre_norm=False,
                 self_attention1=True, self_attention2=True,
                 cross_attention1=True, cross_attention2=True,
                 apply_ffn=True,
                 slot_attention12=False, slot_attention21=False,
                 rotary_pe=False, use_adaln=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.update_seq1 = self_attention1 or cross_attention1
        self.update_seq2 = self_attention2 or cross_attention2
        for _ in range(num_layers):
            self.layers.append(ParallelAttentionLayer(
                d_model=d_model,
                dropout=dropout,
                n_heads=n_heads,
                pre_norm=pre_norm,
                self_attention1=self_attention1,
                self_attention2=self_attention2,
                cross_attention1=cross_attention1,
                cross_attention2=cross_attention2,
                apply_ffn=apply_ffn,
                slot_attention12=slot_attention12,
                slot_attention21=slot_attention21,
                rotary_pe=rotary_pe,
                use_adaln=use_adaln
            ))

    def forward(self, seq1, seq1_key_padding_mask, seq2,
                seq2_key_padding_mask,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                ada_sgnl=None):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        for layer in self.layers:
            seq1_, seq2_ = layer(
                seq1=seq1, seq1_key_padding_mask=seq1_key_padding_mask,
                seq2=seq2, seq2_key_padding_mask=seq2_key_padding_mask,
                seq1_pos=seq1_pos, seq2_pos=seq2_pos,
                seq1_sem_pos=seq1_sem_pos, seq2_sem_pos=seq2_sem_pos,
                ada_sgnl=ada_sgnl
            )
            if self.update_seq1:
                seq1 = seq1_
            if self.update_seq2:
                seq2 = seq2_
        return seq1, seq2


class AdaLN(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
             nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (N, B, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class FeedforwardLayer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.0,
                 use_adaln=False):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, diff_ts=None):
        if diff_ts is not None:
            x = self.adaln(x, diff_ts)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class RelativeCrossAttentionLayer(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.0, use_adaln=False):
        super().__init__()
        self.multihead_attn = Mamba(d_model=embedding_dim)  # Mamba
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None, pad_mask=None):
        if diff_ts is not None:
            adaln_query = self.adaln(query, diff_ts)
        else:
            adaln_query = query
        attn_output = self.multihead_attn(adaln_query, inference_params=None)
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output


class SelfAttentionLayer(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.0, use_adaln=False):
        super().__init__()
        self.multihead_attn = Mamba(d_model=embedding_dim)  # Mamba
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def forward(self, query, diff_ts=None,
                query_pos=None, value_pos=None, pad_mask=None):
        if diff_ts is not None:
            adaln_query = self.adaln(query, diff_ts)
        else:
            adaln_query = query
        attn_output = self.multihead_attn(adaln_query, inference_params=None)
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output


class FFWRelativeCrossAttentionModule(nn.Module):

    def __init__(self, embedding_dim, num_attn_heads, num_layers,
                 use_adaln=True):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(
                embedding_dim, num_attn_heads, use_adaln=use_adaln
            ))
            self.ffw_layers.append(FeedforwardLayer(
                embedding_dim, embedding_dim, use_adaln=use_adaln
            ))

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None):
        output = []
        for i in range(self.num_layers):
            query = self.attn_layers[i](
                query, value, diff_ts, query_pos, value_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output


class FFWRelativeSelfAttentionModule(nn.Module):

    def __init__(self, embedding_dim, num_attn_heads, num_layers,
                 use_adaln=True):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(
                embedding_dim, num_attn_heads, use_adaln=use_adaln
            ))
            self.ffw_layers.append(FeedforwardLayer(
                embedding_dim, embedding_dim, use_adaln=use_adaln
            ))

    def forward(self, query, diff_ts=None,
                query_pos=None, context=None, context_pos=None):
        output = []
        for i in range(self.num_layers):
            query = self.attn_layers[i](
                query, query, diff_ts, query_pos, query_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output


class FFWRelativeSelfCrossAttentionModule(nn.Module):

    def __init__(self, embedding_dim, num_attn_heads,
                 num_self_attn_layers, num_cross_attn_layers, use_adaln=True):
        super().__init__()

        self.num_layers = num_self_attn_layers
        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()

        cross_inds = np.linspace(
            0,
            num_self_attn_layers,
            num_cross_attn_layers + 1,
            dtype=np.int32
        ).tolist()
        for ind in range(num_self_attn_layers):
            self.self_attn_layers.append(RelativeCrossAttentionLayer(
                embedding_dim, num_attn_heads, use_adaln=use_adaln
            ))
            if ind in cross_inds:
                self.cross_attn_layers.append(RelativeCrossAttentionLayer(
                    embedding_dim, num_attn_heads, use_adaln=use_adaln
                ))
            else:
                self.cross_attn_layers.append(None)
            self.ffw_layers.append(FeedforwardLayer(
                embedding_dim, embedding_dim, use_adaln=use_adaln
            ))

    def forward(self, query, context, diff_ts=None,
                query_pos=None, context_pos=None):
        output = []
        for i in range(self.num_layers):
            # Cross attend to the context first
            if self.cross_attn_layers[i] is not None:
                if context_pos is None:
                    cur_query_pos = None
                else:
                    cur_query_pos = query_pos
                query = self.cross_attn_layers[i](
                    query, context, diff_ts, cur_query_pos, context_pos
                )
            # Self attend next
            query = self.self_attn_layers[i](
                query, diff_ts, query_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output
