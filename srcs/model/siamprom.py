# Passion4ever

import math

import numpy as np
import torch
import torch.nn as nn


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SeqEmbedding(nn.Module):
    def __init__(self, src_vocab_size, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, dropout, max_len)

    def forward(self, x):
        seq_emb = self.pos_emb(self.tok_emb(x))

        return seq_emb


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()

        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        scores.masked_fill_(attn_mask, -1e9 if scores.dtype == torch.float32 else -1e4)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()

        self.d_k, self.d_v, self.n_heads = d_k, d_v, n_heads

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.scaled_dot = ScaledDotProductAttention(d_k)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_mat = self.W_Q(Q).view(batch_size, self.n_heads, -1, self.d_k)
        k_mat = self.W_K(K).view(batch_size, self.n_heads, -1, self.d_k)
        v_mat = self.W_V(V).view(batch_size, self.n_heads, -1, self.d_v)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = self.scaled_dot(q_mat, k_mat, v_mat, attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        output = self.linear(context)

        return self.layer_norm(output + residual)


class Convolution1D(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.conv_1d = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        conv_output = self.conv_1d(input.transpose(1, 2)).transpose(1, 2)

        # return self.layer_norm(conv_output + input)
        return conv_output + input  


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.bilstm = nn.Sequential(
            nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = nn.Linear(2 * hidden_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, input):
        lstm_output, (hn, cn) = self.bilstm(input)
        lstm_output = self.linear(lstm_output)

        return self.layer_norm(lstm_output + input)


class Compressor(nn.Module):
    def __init__(self, d_model, shape_1, shape_2):
        super().__init__()

        self.compressor = nn.Sequential(
            nn.Linear(3 * d_model, shape_1),
            nn.LayerNorm(shape_1),
            nn.LeakyReLU(),
            nn.Linear(shape_1, shape_2),
        )

    def forward(self, attn_out, conv_out, lstm_out):
        attn_out = attn_out.mean(dim=1)
        conv_out = conv_out.mean(dim=1)
        lstm_out = lstm_out.mean(dim=1)
        input = torch.concat([attn_out, conv_out, lstm_out], dim=1)
        output = self.compressor(input)

        return output


class PromRepresentationNet(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        d_model,
        d_k,
        d_v,
        n_heads,
        d_ff,
        d_hidden,
        num_layers,
        shape_1,
        shape_2,
    ):
        super().__init__()

        self.seq_emb = SeqEmbedding(src_vocab_size, d_model)
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.conv_1d = Convolution1D(d_model, d_ff)
        self.bilstm = BiLSTM(d_model, d_hidden, num_layers)
        self.compressor = Compressor(d_model, shape_1, shape_2)

    def forward(self, input):
        embed_out = self.seq_emb(input)
        self_attn_mask = get_attn_pad_mask(input, input)
        attn_out = self.self_attn(embed_out, embed_out, embed_out, self_attn_mask)
        conv_out = self.conv_1d(embed_out)
        lstm_out = self.bilstm(embed_out)

        seq_representation = self.compressor(attn_out, conv_out, lstm_out)

        return seq_representation


class SiamProm(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        d_model,
        d_k,
        d_v,
        n_heads,
        d_ff,
        d_hidden,
        n_layers,
        shape_1,
        shape_2,
        shape_3,
    ):
        super().__init__()

        self.siamese_net = PromRepresentationNet(
            src_vocab_size,
            d_model,
            d_k,
            d_v,
            n_heads,
            d_ff,
            d_hidden,
            n_layers,
            shape_1,
            shape_2,
        )

        self.predictor = nn.Sequential(
            nn.LayerNorm(shape_2),
            nn.LeakyReLU(),
            nn.Linear(shape_2, shape_3),
            nn.LayerNorm(shape_3),
            nn.LeakyReLU(),
            nn.Linear(shape_3, 2),
        )

    def forward(self, input):
        seq_fea_vec = self.siamese_net(input)

        return seq_fea_vec

    def predict(self, input):
        with torch.no_grad():
            output = self.forward(input)
        return self.predictor(output)
