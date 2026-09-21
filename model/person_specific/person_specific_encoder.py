import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # device 인자 제거하고 register_buffer 사용
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))

        # ← 핵심: register_buffer로 등록하면 DataParallel이 자동으로 device 이동
        self.register_buffer('encoding', pe)

    def forward(self, x):
        encoding = self.encoding[:x.shape[1], :]   # [seq_len, d_model]
        return encoding.unsqueeze(0)               # [1, seq_len, d_model]


class Transformer(nn.Module):
    def __init__(self, device, in_features, embed_dim, num_heads, num_layers, mlp_dim, seq_len, proj_dim,
                 proj_head="mlp", drop_prob=0.1, max_len=5000, pos_encoding="absolute", embed_layer="linear"):
        super(Transformer, self).__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.seq_len = seq_len
        self.proj_dim = proj_dim
        self.max_len = max_len
        self.embed_dim = embed_dim if embed_layer == "linear" else in_features
        self.embed_layer = nn.Linear(in_features, embed_dim) if embed_layer == "linear" else nn.Identity()

        self.pos_encoding = pos_encoding
        if pos_encoding == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.seq_len, self.embed_dim))
        elif pos_encoding == "absolute":
            self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=max_len, device=device)
        else:
            raise NotImplementedError('position encoding method not supported: {}'.format(pos_encoding))

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # 주석처리1
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.mlp_dim,
                batch_first=True),
            num_layers
        )  # encoder

        self.dropout = nn.Dropout(p=drop_prob) # dropout layer
        if proj_head == 'linear':
            self.proj_head = nn.Linear(embed_dim, proj_dim)
        elif proj_head == 'mlp':
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, proj_dim)
            )
        else:
            self.proj_head = nn.Identity()

    def forward(self, x, padding_mask=None):
        x = self.embed_layer(x)
        # x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) # 주석처리2
        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "learnable":
            x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # ── 수정 시작 ──
        feat = x  # [B, T, 512] 전체 시퀀스
        proj = F.normalize(self.proj_head(feat), dim=-1)  # [B, T, proj_dim] frame-wise
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).to(feat.dtype)
            feat = feat * valid
            proj = proj * valid
        # ── 수정 끝 ──

        return feat, proj
