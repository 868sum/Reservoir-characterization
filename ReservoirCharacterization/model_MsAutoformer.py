import torch
import torch.nn as nn
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import EncoderLayer, my_Layernorm

class MsAutoformer_EncoderOnly(nn.Module):
    def __init__(self, configs):
        super(MsAutoformer_EncoderOnly, self).__init__()
        self.seq_len = configs.seq_len
        self.output_dim = configs.output_size
        self.d_model = configs.hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(configs.input_size, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(configs.dropout),

            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )

        self.cnn_norm = nn.LayerNorm(self.d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(configs.input_size * configs.seq_len, configs.hidden_space),
            nn.ReLU(),
            nn.LayerNorm(configs.hidden_space)
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    self.d_model, configs.num_heads
                ),
                self.d_model,
                configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation
            )
            for _ in range(configs.e_layers)
        ])

        self.total_d_model = configs.e_layers * self.d_model + configs.hidden_space
        self.residual_projector = nn.Sequential(
            nn.Linear(self.total_d_model, configs.hidden_space),
            nn.ReLU(),
            nn.LayerNorm(configs.hidden_space),
            nn.Linear(configs.hidden_space, self.output_dim)
        )

    def forward(self, x_enc):

        B, L, C = x_enc.shape

        cnn_out = self.cnn(x_enc.permute(0, 2, 1))
        cnn_out = cnn_out.permute(0, 2, 1)
        cnn_out = self.cnn_norm(cnn_out)

        enc_out = cnn_out
        layer_outputs = []
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out, attn_mask=None)
            layer_outputs.append(enc_out)


        concat_out = torch.cat(layer_outputs, dim=-1)
        pooled_out = concat_out[:, -1, :]
        input_flat = x_enc.reshape(B, -1)
        input_embed = self.input_proj(input_flat)
        final_rep = torch.cat([pooled_out, input_embed], dim=-1)


        out = self.residual_projector(final_rep)
        return out.squeeze(-1)