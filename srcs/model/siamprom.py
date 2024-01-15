# Passion4ever

import math

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Parameter
import mindspore.numpy as mnp



def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.equal(0).unsqueeze(1)
    return pad_attn_mask.broadcast_to((batch_size, len_q, len_k))


class PositionEmbedding(nn.Cell):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = ops.zeros((max_len, d_model))
        position = mnp.arange(0, max_len, dtype=ms.float32).unsqueeze(1)
        div_term = ops.exp(
            mnp.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = ops.sin(position * div_term)
        pe[:, 1::2] = ops.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.pe = Parameter(pe, name="pe", requires_grad=False)

    def construct(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    

class SeqEmbedding(nn.Cell):
    def __init__(self, src_vocab_size, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, dropout, max_len)

    def construct(self, x):
        seq_emb = self.pos_emb(self.tok_emb(x))

        return seq_emb


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k):
        super().__init__()

        self.d_k = d_k

    def construct(self, Q, K, V, attn_mask):
        scores = ops.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill(attn_mask, -1e9 if scores.dtype == ms.float32 else -1e4)
        attn = nn.Softmax(axis=-1)(scores)
        context = ops.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()

        self.d_k, self.d_v, self.n_heads = d_k, d_v, n_heads
        self.W_Q = nn.Dense(d_model, d_k * n_heads)
        self.W_K = nn.Dense(d_model, d_k * n_heads)
        self.W_V = nn.Dense(d_model, d_v * n_heads)
        self.scaled_dot = ScaledDotProductAttention(d_k)
        self.linear = nn.Dense(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm([d_model])

    def construct(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.shape[0]

        q_mat = self.W_Q(Q).view(batch_size, self.n_heads, -1, self.d_k)
        k_mat = self.W_K(K).view(batch_size, self.n_heads, -1, self.d_k)
        v_mat = self.W_V(V).view(batch_size, self.n_heads, -1, self.d_v)

        attn_mask = attn_mask.unsqueeze(1).tile((1, self.n_heads, 1, 1))

        context, attn = self.scaled_dot(q_mat, k_mat, v_mat, attn_mask)
        
        context = (
            context.swapaxes(1, 2)
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        
        output = self.linear(context)
        return self.layer_norm(output + residual)



class Convolution1D(nn.Cell):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.conv_1d = nn.SequentialCell(
            nn.Conv1d(d_model, d_ff, kernel_size=3, pad_mode="same", has_bias=True),
            nn.LeakyReLU(),
            nn.Conv1d(d_ff, d_model, kernel_size=1, has_bias=True),
        )
        self.layer_norm = nn.LayerNorm([d_model])

    def construct(self, input):
        conv_output = self.conv_1d(input.swapaxes(1, 2)).swapaxes(1, 2)

        # return self.layer_norm(conv_output + input)
        return conv_output + input  


class BiLSTM(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.bilstm = nn.SequentialCell(
            nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = nn.Dense(2 * hidden_size, input_size)
        self.layer_norm = nn.LayerNorm([input_size])

    def construct(self, input):
        lstm_output, (hn, cn) = self.bilstm(input)
        lstm_output = self.linear(lstm_output)

        return self.layer_norm(lstm_output + input)


class Compressor(nn.Cell):
    def __init__(self, d_model, shape_1, shape_2):
        super().__init__()

        self.compressor = nn.SequentialCell(
            nn.Dense(3 * d_model, shape_1),
            nn.LayerNorm([shape_1]),
            nn.LeakyReLU(),
            nn.Dense(shape_1, shape_2),
        )

    def construct(self, attn_out, conv_out, lstm_out):
        attn_out = attn_out.mean(axis=1)
        conv_out = conv_out.mean(axis=1)
        lstm_out = lstm_out.mean(axis=1)
        input = ops.concat([attn_out, conv_out, lstm_out], axis=1)
        output = self.compressor(input)

        return output


class PromRepresentationNet(nn.Cell):
    def __init__(self, src_vocab_size, d_model, d_k, d_v, n_heads, d_ff, d_hidden, num_layers, shape_1, shape_2):
        super().__init__()

        self.seq_emb = SeqEmbedding(src_vocab_size, d_model)
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.conv_1d = Convolution1D(d_model, d_ff)
        self.bilstm = BiLSTM(d_model, d_hidden, num_layers)
        self.compressor = Compressor(d_model, shape_1, shape_2)

    def construct(self, input):
        embed_out = self.seq_emb(input)
        self_attn_mask = get_attn_pad_mask(input, input)
        attn_out = self.self_attn(embed_out, embed_out, embed_out, self_attn_mask)
        conv_out = self.conv_1d(embed_out)
        lstm_out = self.bilstm(embed_out)

        seq_representation = self.compressor(attn_out, conv_out, lstm_out)

        return seq_representation


class SiamProm(nn.Cell):
    def __init__(self, src_vocab_size, d_model, d_k, d_v, n_heads, d_ff, d_hidden, n_layers, shape_1, shape_2, shape_3):
        super().__init__()

        self.siamese_net = PromRepresentationNet(src_vocab_size, d_model, 
                                                 d_k, d_v, n_heads, d_ff, 
                                                 d_hidden, n_layers, shape_1, shape_2)

        self.predictor = nn.SequentialCell(
            nn.LayerNorm([shape_2]),
            nn.LeakyReLU(),
            nn.Dense(shape_2, shape_3),
            nn.LayerNorm([shape_3]),
            nn.LeakyReLU(),
            nn.Dense(shape_3, 2),
        )

    def construct(self, input):
        seq_fea_vec = self.siamese_net(input)

        return seq_fea_vec

    def predict(self, input):
        output = ops.stop_gradient(self.construct(input))


        return self.predictor(output)
