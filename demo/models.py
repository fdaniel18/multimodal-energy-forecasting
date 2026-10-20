from __future__ import annotations
import os
import sys
import torch
import torch.nn as nn

FORECAST = 24
LOOKBACK = 168


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from sota_baselines import PatchTST, PatchTSTConfig  # noqa: E402


class BaselineLSTM(nn.Module):
    def __init__(self, num_math_features, hidden_dim=64, forecast=FORECAST):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_math_features, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, forecast),
        )

    def forward(self, math_input):
        _, (h, _) = self.lstm(math_input)
        math_vector = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(math_vector)


class GlobalTextConditionedModel(nn.Module):
    def __init__(self, num_math_features, text_emb_dim=768, hidden_dim=64,
                 num_building_types=16, forecast=FORECAST):
        super().__init__()
        math_out_dim = hidden_dim * 2

        self.lstm = nn.LSTM(input_size=num_math_features, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.static_processor = nn.Sequential(
            nn.Linear(text_emb_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, math_out_dim), nn.LayerNorm(math_out_dim),
        )
        self.dynamic_processor = nn.Sequential(
            nn.Linear(text_emb_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, math_out_dim), nn.LayerNorm(math_out_dim),
        )
        self.film = nn.Linear(math_out_dim, math_out_dim * 2)
        self.math_head = nn.Sequential(
            nn.Linear(math_out_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, forecast),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(math_out_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, forecast),
        )
        self.building_classifier = nn.Linear(math_out_dim, num_building_types)
        self._attn_scale = math_out_dim ** 0.5

    def _attention(self, query, lstm_seq):
        scores = torch.bmm(lstm_seq, query.unsqueeze(-1)).squeeze(-1)
        weights = torch.softmax(scores / self._attn_scale, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_seq).squeeze(1)
        return context, weights

    def forward(self, math_input, static_emb, dynamic_emb, return_attn=False):
        lstm_seq, (h_n, _) = self.lstm(math_input)
        math_vector = torch.cat((h_n[-2], h_n[-1]), dim=1)

        static_vec = self.static_processor(static_emb)
        dynamic_vec = self.dynamic_processor(dynamic_emb)

        gamma, beta = self.film(static_vec).chunk(2, dim=-1)
        math_cond = gamma * math_vector + beta
        math_pred = self.math_head(math_cond)

        context, attn = self._attention(dynamic_vec, lstm_seq)
        offset = self.offset_head(context)

        pred = math_pred + offset
        building_logits = self.building_classifier(static_vec)
        if return_attn:
            return pred, building_logits, math_pred, offset, attn
        return pred, building_logits, math_pred, offset


def build_patchtst(num_math_features, seq_len=LOOKBACK, pred_len=FORECAST):
    cfg = PatchTSTConfig(seq_len=seq_len, pred_len=pred_len, enc_in=num_math_features)
    return PatchTST(cfg)


def load_model(kind: str, checkpoint_path: str, num_math_features: int,
               device="cpu", num_building_types: int = 16):
    if kind == "notext":
        model = BaselineLSTM(num_math_features)
    elif kind == "hybrid":
        model = GlobalTextConditionedModel(num_math_features,
                                           num_building_types=num_building_types)
    elif kind == "patchtst":
        model = build_patchtst(num_math_features)
    else:
        raise ValueError(f"kind necunoscut: {kind}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
