import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# PerFRDiff 내부 모듈 import
from model.diffusion.rnn import AutoencoderRNN_VAE_v2
from model.person_specific.person_specific_encoder import Transformer as PersonTransformer

logger = logging.getLogger(__name__)


def _normalize_lengths(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    if lengths is None:
        return torch.full(
            (x.shape[0],), x.shape[1], device=x.device, dtype=torch.long
        )
    return lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.shape[1])


def _valid_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _resample_valid_batch(x: torch.Tensor, source_lengths: torch.Tensor,
                          target_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly resample each valid prefix and return padded output plus mask."""
    max_target = int(target_lengths.max().item())
    rows = []
    for index in range(x.shape[0]):
        source_len = int(source_lengths[index].item())
        target_len = int(target_lengths[index].item())
        valid = x[index:index + 1, :source_len].transpose(1, 2)
        resized = F.interpolate(
            valid, size=target_len, mode="linear", align_corners=False
        ).transpose(1, 2).squeeze(0)
        if target_len < max_target:
            resized = F.pad(resized, (0, 0, 0, max_target - target_len))
        rows.append(resized)
    output = torch.stack(rows, dim=0)
    mask = _valid_mask(target_lengths, max_target)
    return output * mask.unsqueeze(-1).to(output.dtype), mask

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: str, model: nn.Module, strict: bool = True,
                     tag: str = "") -> dict:
    """
    체크포인트를 로드하고 결과를 상세히 로깅한다.
    Returns: load_result dict (missing_keys, unexpected_keys 포함)
    """
    assert os.path.exists(ckpt_path), f"[SBE] 체크포인트 없음: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    result = model.load_state_dict(state_dict, strict=strict)

    missing   = result.missing_keys
    unexpected = result.unexpected_keys
    logger.info(f"[SBE][{tag}] 체크포인트 로드: {ckpt_path}")
    logger.info(f"  missing_keys   ({len(missing)}): "
                f"{missing[:5]}{'...' if len(missing)>5 else ''}")
    logger.info(f"  unexpected_keys({len(unexpected)}): "
                f"{unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    if strict and (missing or unexpected):
        raise RuntimeError(f"[SBE][{tag}] strict 로드 실패 — 위 키 확인 필요")
    return {"missing": missing, "unexpected": unexpected}


def _freeze(module: nn.Module, tag: str = ""):
    """모듈의 모든 파라미터를 freeze."""
    for p in module.parameters():
        p.requires_grad = False
    logger.info(f"[SBE] freeze: {tag}")


def _unfreeze(module: nn.Module, tag: str = ""):
    """모듈의 모든 파라미터를 unfreeze."""
    for p in module.parameters():
        p.requires_grad = True
    logger.info(f"[SBE] unfreeze: {tag}")


# ──────────────────────────────────────────────────────────────────────────────
# Branch 1: mel 오디오 인코더  (80-dim → 512-dim)
# ──────────────────────────────────────────────────────────────────────────────

class MelAudioEncoder(nn.Module):
    """
    mel [B, T, 80] → E_aud [B, T, 512]   ← 시간축 보존
    """

    def __init__(self, mel_dim: int = 80):
        super().__init__()
        self.mel_proj = nn.Linear(mel_dim, 512)
        self.norm = nn.LayerNorm(512)

    def forward(self, mel: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            mel: [B, T, 80]
        Returns:
            E_aud: [B, T, 512]
        """
        x = self.mel_proj(mel)   # [B, T, 512]
        x = self.norm(x)
        # x.mean(dim=1) 삭제
        lengths = _normalize_lengths(mel, lengths)
        return x * _valid_mask(lengths, x.shape[1]).unsqueeze(-1).to(x.dtype)


class MfccRefEncoder(nn.Module):
    """
    Stage-1 alignment 전용 teacher 인코더. (학습 후 제거 가능)
    MFCC [B, T, 78] → ref_features [B, 512]  (항상 freeze)

    DiffusionPriorNetwork 의 to_audio_encodings (Linear 78→512) 가중치를 재활용.
    해당 ckpt 가 없으면 random 초기화된 채로 freeze — 그래도 "고정된 타겟 공간"으로 기능.
    """

    def __init__(self, mfcc_dim: int = 78):
        super().__init__()
        self.mfcc_proj = nn.Linear(mfcc_dim, 512)
        self.norm = nn.LayerNorm(512)
        _freeze(self, tag="MfccRefEncoder")

    def load_from_prior_ckpt(self, prior_ckpt_path: str):
        """
        DiffusionPriorNetwork 체크포인트에서 to_audio_encodings 가중치만 추출.
        PerFRDiff ckpt 경로: checkpoints/diffusion_model/DiffusionPriorNetwork/checkpoint.pth
        """
        if not os.path.exists(prior_ckpt_path):
            logger.warning(f"[SBE][MfccRef] prior ckpt 없음 → random init 유지: {prior_ckpt_path}")
            return

        ckpt = torch.load(prior_ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # DiffusionPriorNetwork 키 예시: "to_audio_encodings.weight"
        w_key = "to_audio_encodings.weight"
        b_key = "to_audio_encodings.bias"
        if w_key in sd:
            with torch.no_grad():
                self.mfcc_proj.weight.copy_(sd[w_key])
                if b_key in sd:
                    self.mfcc_proj.bias.copy_(sd[b_key])
            logger.info(f"[SBE][MfccRef] to_audio_encodings 가중치 로드 성공: {prior_ckpt_path}")
        else:
            logger.warning(f"[SBE][MfccRef] '{w_key}' 키 없음 → random init 유지. "
                           f"(사용 가능한 키: {[k for k in sd if 'audio' in k][:5]})")

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mfcc: [B, T, 78]
        Returns:
            ref_feat: [B, 512]  (gradient 차단)
        """
        with torch.no_grad():
            x = self.mfcc_proj(mfcc)
            x = self.norm(x)
            x = x.mean(dim=1)   # [B, 512]
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Branch 2: 3DMM 외형 인코더  (486-dim → 512-dim)
# ──────────────────────────────────────────────────────────────────────────────

class AppearanceEncoder(nn.Module):
    """
    3DMM [B, T, 486] → E_app [B, T, 512]   ← 시간축 보존
    """

    def __init__(self, dmm_dim: int = 486, embed_dim: int = 512,
                 num_heads: int = 4, num_layers: int = 4,
                 mlp_dim: int = 1024, proj_dim: int = 512,
                 drop_prob: float = 0.1, max_len: int = 2000,
                 device: str = "cpu"):
        super().__init__()

        self.transformer = PersonTransformer(
            device=device,
            in_features=dmm_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            seq_len=750,
            proj_dim=proj_dim,
            proj_head="mlp",
            drop_prob=drop_prob,
            max_len=max_len,
            pos_encoding="absolute",
            embed_layer="linear",
        )
    def load_pretrained_except_embed(self, ckpt_path: str):
        """
        PersonSpecificEncoder 체크포인트에서 embed_layer 제외한 모든 가중치를 로드.

        embed_layer (Linear 58→512) 는 dim 불일치 → skip.
        Transformer 레이어, norm, cls_token, pos_embed 는 pretrained 유지.
        """
        if not os.path.exists(ckpt_path):
            logger.warning(f"[SBE][AppEnc] ckpt 없음 → random init: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # PersonTransformer 의 내부 키에 맞게 prefix 추가
        # (AppearanceEncoder.transformer 아래에 위치)
        own_sd = self.transformer.state_dict()
        filtered = {}
        skipped  = []
        for k, v in sd.items():
            if k not in own_sd:
                skipped.append(k)
                continue
            if "embed_layer" in k:
                # dim 불일치 → skip, 새로 학습
                skipped.append(f"[dim mismatch, skip] {k}: ckpt {v.shape} vs model {own_sd[k].shape}")
                continue
            if own_sd[k].shape != v.shape:
                skipped.append(f"[shape mismatch] {k}: ckpt {v.shape} vs model {own_sd[k].shape}")
                continue
            filtered[k] = v

        result = self.transformer.load_state_dict(filtered, strict=False)
        loaded_n = len(filtered)
        total_n  = len(own_sd)
        logger.info(f"[SBE][AppEnc] pretrained 가중치 로드: {loaded_n}/{total_n} keys")
        logger.info(f"  skip된 키 ({len(skipped)}):")
        for s in skipped[:8]:
            logger.info(f"    {s}")
        if len(skipped) > 8:
            logger.info(f"    ... (총 {len(skipped)}개)")
        logger.info(f"  missing  : {result.missing_keys[:5]}")
        logger.info(f"  unexpected: {result.unexpected_keys[:5]}")

    def forward(self, dmm: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            dmm: [B, T, 486]
        Returns:
            E_app: [B, T, 512]
        """
        # PersonTransformer.forward(x) 가 (feat, proj) 를 반환
        # feat = [B, T, 512] 전체 시퀀스, proj = frame-wise 정규화 (사용 안 함)
        lengths = _normalize_lengths(dmm, lengths)
        padding_mask = ~_valid_mask(lengths, dmm.shape[1])
        feat, _ = self.transformer(dmm, padding_mask=padding_mask)
        return feat  # [B, T, 512]


# ──────────────────────────────────────────────────────────────────────────────
# Branch 3: 감정/AU 인코더  (25-dim → 512-dim)
# ──────────────────────────────────────────────────────────────────────────────

class EmotionEncoder(nn.Module):
    """
    AU [B, T, 25] → E_emo [B, T, 512]   ← 시간축 보존
    """

    def __init__(self, emotion_dim: int = 25, hidden_dim: int = 512,
                 z_dim: int = 512, window_size: int = 50):
        super().__init__()

        from types import SimpleNamespace
        cfg = SimpleNamespace(
            seq_len=750,
            window_size=window_size,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            emb_dims=[128, 128],
            num_layers=2,
            rnn_type="gru",
            dropout=0.0,
            emotion_dim=emotion_dim,
            coeff_3dmm_dim=58,
        )
        self.rnn_encoder = AutoencoderRNN_VAE_v2(cfg)
        self.window_size = window_size
        self.z_dim = z_dim

        # RNN hidden → z_dim 매핑 (frame-wise)
        self.frame_proj = nn.Linear(hidden_dim, z_dim)

    def load_pretrained(self, ckpt_path: str):
        """
        AutoencoderRNN_VAE_v2 사전학습 가중치 로드.
        frame_proj (새로 추가한 레이어) 는 미존재 키이므로 strict=False 로 로드.
        """
        import os, logging
        logger = logging.getLogger(__name__)

        if not os.path.exists(ckpt_path):
            logger.warning(f"[SBE][EmoEnc] ckpt 없음 → random init 유지: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # rnn_encoder 서브모듈에만 로드 (frame_proj 는 새로 학습)
        own_sd = self.rnn_encoder.state_dict()
        filtered = {k: v for k, v in sd.items()
                    if k in own_sd and own_sd[k].shape == v.shape}
        skipped = [k for k in sd if k not in filtered]

        result = self.rnn_encoder.load_state_dict(filtered, strict=False)
        logger.info(f"[SBE][EmoEnc] pretrained 로드: {len(filtered)}/{len(own_sd)} keys")
        logger.info(f"  skip ({len(skipped)}): {skipped[:5]}")
        logger.info(f"  missing   : {result.missing_keys[:3]}")
        logger.info(f"  unexpected: {result.unexpected_keys[:3]}")

    def encode_variable(self, au: torch.Tensor,
                        lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            au: [B, T, 25]
        Returns:
            z_seq: [B, T, 512]   ← 변경: 전체 시퀀스 반환
        """
        B, T, D = au.shape
        lengths = _normalize_lengths(au, lengths)
        au = au * _valid_mask(lengths, T).unsqueeze(-1).to(au.dtype)
        au_seq = rearrange(au, "b t d -> t b d")   # [T, B, 25]

        h = self.rnn_encoder.x_rnn(au_seq)         # [T, B, hidden_dim]
        h = rearrange(h, "t b d -> b t d")         # [B, T, hidden_dim]

        # 매 프레임마다 mu 생성 (기존: 마지막 프레임만)
        z_seq = self.frame_proj(h)                  # [B, T, 512]
        return z_seq * _valid_mask(lengths, T).unsqueeze(-1).to(z_seq.dtype)

    def forward(self, au: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_variable(au, lengths)


# ──────────────────────────────────────────────────────────────────────────────
# Fusion MLP
# ──────────────────────────────────────────────────────────────────────────────

import torch.nn.functional as F


class FusionMLP(nn.Module):
    """
    [E_aud | E_app | E_emo] concat → c [B, T, 512]

    구조:
        시간축 정렬 → concat → Linear(1536→512) → ReLU → Linear(512→512)
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, e_aud: torch.Tensor,
                e_app: torch.Tensor,
                e_emo: torch.Tensor,
                audio_lengths: torch.Tensor,
                app_lengths: torch.Tensor,
                emotion_lengths: torch.Tensor,
                target_lengths: torch.Tensor):
        """
        Args:
            e_aud: [B, T1, 512]
            e_app: [B, T2, 512]
            e_emo: [B, T3, 512]
        Returns:
            c: [B, T, 512]   (T = min(T1, T2, T3))
        """
        e_aud, mask = _resample_valid_batch(e_aud, audio_lengths, target_lengths)
        e_app, _ = _resample_valid_batch(e_app, app_lengths, target_lengths)
        e_emo, _ = _resample_valid_batch(e_emo, emotion_lengths, target_lengths)

        x = torch.cat([e_aud, e_app, e_emo], dim=-1)  # [B, T, 1536]
        c = self.net(x) * mask.unsqueeze(-1).to(x.dtype)
        return c, mask, (e_aud, e_app, e_emo)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 SBE 클래스
# ──────────────────────────────────────────────────────────────────────────────

class SpeakerBehaviorEncoder(nn.Module):
    """
    전체 SBE 모듈.

    사용 예시:
        # 빌드 + 사전학습 가중치 로드
        sbe = SpeakerBehaviorEncoder.from_pretrained(
            ckpt_dir    = "checkpoints",
            prior_ckpt  = "checkpoints/diffusion_model/DiffusionPriorNetwork/checkpoint.pth",
            device      = device,
        )

        # Stage-1 (alignment) 모드
        sbe.set_stage("align")
        loss_align = sbe.alignment_loss(mel, mfcc)

        # Stage-2 (main) 모드
        sbe.set_stage("main")
        c = sbe(mel, dmm, au)   # [B, 512]
    """

    STAGES = ("align", "main")

    def __init__(self, device: str = "cpu",
                 mel_dim: int = 80, mfcc_dim: int = 78,
                 dmm_dim: int = 486, au_dim: int = 25,
                 embed_dim: int = 512,
                 # AppearanceEncoder 하이퍼파라미터
                 app_num_heads: int = 4, app_num_layers: int = 4,
                 app_mlp_dim: int = 1024, app_max_len: int = 2000,
                 window_size: int = 50,
                 mel_frame_hz: float = 22050 / 256,
                 video_frame_hz: float = 25.0,
                 target_frame_hz: float = 25.0):
        super().__init__()

        # ── Branch 1: 오디오 ──────────────────────────────────────────────
        self.mel_encoder  = MelAudioEncoder(mel_dim=mel_dim)
        self.mfcc_ref_enc = MfccRefEncoder(mfcc_dim=mfcc_dim)  # Stage-1 teacher

        # ── Branch 2: 외형 (3DMM) ─────────────────────────────────────────
        self.app_encoder = AppearanceEncoder(
            dmm_dim=dmm_dim, embed_dim=embed_dim,
            num_heads=app_num_heads, num_layers=app_num_layers,
            mlp_dim=app_mlp_dim, proj_dim=embed_dim,
            max_len=app_max_len, device=device,
        )

        # ── Branch 3: 감정/AU ─────────────────────────────────────────────
        self.emo_encoder = EmotionEncoder(
            emotion_dim=au_dim, hidden_dim=embed_dim,
            z_dim=embed_dim, window_size=window_size,
        )

        # ── Fusion MLP ────────────────────────────────────────────────────
        self.fusion_mlp = FusionMLP(embed_dim=embed_dim)
        self.mel_frame_hz = float(mel_frame_hz)
        self.video_frame_hz = float(video_frame_hz)
        self.target_frame_hz = float(target_frame_hz)

        self._stage = "main"  # 기본값

        logger.info(
            f"[SBE] 초기화 완료 | mel:{mel_dim}d  mfcc:{mfcc_dim}d  "
            f"3dmm:{dmm_dim}d  AU:{au_dim}d  embed:{embed_dim}d"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 사전학습 가중치 로드
    # ──────────────────────────────────────────────────────────────────────

    def load_pretrained_weights(self,
                                ckpt_dir: str,
                                prior_ckpt_path: str = None):
        """
        각 branch 의 사전학습 가중치를 개별 로드.

        Args:
            ckpt_dir       : PerFRDiff checkpoints/ 폴더
            prior_ckpt_path: DiffusionPriorNetwork ckpt (Stage-1 MFCC ref용)
        """
        logger.info("=" * 60)
        logger.info("[SBE] 사전학습 가중치 로드 시작")
        logger.info("=" * 60)

        # (1) Stage-1 teacher: MFCC ref
        if prior_ckpt_path:
            logger.info("[SBE] [1/3] MFCC Ref 인코더 (Stage-1 teacher)")
            self.mfcc_ref_enc.load_from_prior_ckpt(prior_ckpt_path)
        else:
            logger.info("[SBE] [1/3] MFCC Ref — prior_ckpt_path 미지정, random init 사용")

        # (2) AppearanceEncoder: embed_layer 제외 pretrained
        app_ckpt = os.path.join(ckpt_dir, "person_specific", "checkpoint.pth")
        logger.info("[SBE] [2/3] Appearance Encoder (PersonSpecificEncoder)")
        self.app_encoder.load_pretrained_except_embed(app_ckpt)

        # (3) EmotionEncoder: strict load
        emo_ckpt = os.path.join(ckpt_dir, "embedder_latent", "checkpoint.pth")
        logger.info("[SBE] [3/3] Emotion Encoder (AutoencoderRNN_VAE_v2)")
        self.emo_encoder.load_pretrained(emo_ckpt)

        logger.info("[SBE] 사전학습 가중치 로드 완료")
        logger.info("=" * 60)

    @classmethod
    def from_pretrained(cls, ckpt_dir: str,
                        prior_ckpt_path: str = None,
                        device: str = "cpu", **kwargs):
        """
        SBE 를 생성하고 바로 사전학습 가중치를 로드한다.

        Args:
            ckpt_dir       : PerFRDiff checkpoints/ 루트 폴더
            prior_ckpt_path: DiffusionPriorNetwork ckpt 경로 (optional)
            device         : 'cpu' or 'cuda:0'
            **kwargs       : SpeakerBehaviorEncoder.__init__ 추가 인자
        Returns:
            SpeakerBehaviorEncoder instance
        """
        logger.info(f"[SBE] from_pretrained | ckpt_dir={ckpt_dir} | device={device}")
        sbe = cls(device=device, **kwargs)
        sbe.load_pretrained_weights(ckpt_dir=ckpt_dir,
                                    prior_ckpt_path=prior_ckpt_path)
        sbe = sbe.to(device)
        logger.info(f"[SBE] 모델을 {device} 로 이동 완료")
        return sbe

    # ──────────────────────────────────────────────────────────────────────
    # 학습 Stage 전환
    # ──────────────────────────────────────────────────────────────────────

    def set_stage(self, stage: str):
        """
        stage="align"  → mel_encoder 만 학습, 나머지 freeze
        stage="main"   → 전체 학습 가능 (mfcc_ref_enc 는 항상 freeze)
        """
        assert stage in self.STAGES, f"stage 는 {self.STAGES} 중 하나여야 합니다."
        self._stage = stage

        if stage == "align":
            # mel_encoder 만 학습
            _unfreeze(self.mel_encoder, "mel_encoder")
            _freeze(self.mfcc_ref_enc, "mfcc_ref_enc")
            _freeze(self.app_encoder,  "app_encoder")
            _freeze(self.emo_encoder,  "emo_encoder")
            _freeze(self.fusion_mlp,   "fusion_mlp")
            logger.info("[SBE] Stage=align | mel_encoder 학습, 나머지 freeze")

        elif stage == "main":
            _unfreeze(self.mel_encoder, "mel_encoder")
            _freeze(self.mfcc_ref_enc,  "mfcc_ref_enc")   # teacher는 항상 freeze
            _unfreeze(self.app_encoder, "app_encoder")
            _unfreeze(self.emo_encoder, "emo_encoder")
            _unfreeze(self.fusion_mlp,  "fusion_mlp")
            logger.info("[SBE] Stage=main | mfcc_ref_enc 제외 전체 학습")

    # ──────────────────────────────────────────────────────────────────────
    # Stage-1 Alignment Loss
    # ──────────────────────────────────────────────────────────────────────

    def alignment_loss(self, mel: torch.Tensor, mfcc: torch.Tensor,
                       mel_len: torch.Tensor | None = None,
                       mfcc_len: torch.Tensor | None = None) -> torch.Tensor:
        """
        Stage-1 학습용 MSE alignment loss.

        mel 과 mfcc 의 시간 길이(T)가 다를 수 있으므로 min(T_mel, T_mfcc) 로 자른다.

        Args:
            mel  : [B, T_mel,  80]
            mfcc : [B, T_mfcc, 78]
        Returns:
            loss : scalar tensor
        """
        assert self._stage == "align", "alignment_loss 는 stage='align' 에서만 사용."

        mel_len = _normalize_lengths(mel, mel_len)
        mfcc_len = _normalize_lengths(mfcc, mfcc_len)
        target_len = torch.minimum(mel_len, mfcc_len)

        mel_feat = self.mel_encoder.norm(self.mel_encoder.mel_proj(mel))

        # teacher: MFCC ref (detach, no grad)
        with torch.no_grad():
            mfcc_feat = self.mfcc_ref_enc.norm(
                self.mfcc_ref_enc.mfcc_proj(mfcc)
            )
        mel_feat, mask = _resample_valid_batch(mel_feat, mel_len, target_len)
        mfcc_feat, _ = _resample_valid_batch(mfcc_feat, mfcc_len, target_len)
        error = (mel_feat - mfcc_feat).pow(2) * mask.unsqueeze(-1)
        denom = (mask.sum() * mel_feat.shape[-1]).clamp(min=1)
        loss = error.sum() / denom
        return loss

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, mel: torch.Tensor,
                dmm: torch.Tensor,
                au: torch.Tensor,
                mel_len: torch.Tensor | None = None,
                dmm_len: torch.Tensor | None = None,
                au_len: torch.Tensor | None = None,
                return_branches: bool = False,
                return_mask: bool = False):
        """
        전체 SBE forward pass. Stage-2(main) 에서 사용.

        Args:
            mel  : [B, T_mel, 80]   — Speaker A 의 mel-spectrogram
            dmm  : [B, T_dmm, 486]  — Speaker A 의 3DMM 계수 (per-frame stack)
            au   : [B, T_au,  25]   — Speaker A 의 AU (Action Unit)
            return_branches: True 이면 E_aud, E_app, E_emo 도 함께 반환

        Returns:
            c           : [B, 512]  — context embedding (downstream decoder 조건)
            (optional) E_aud, E_app, E_emo : each [B, 512]
        """
        mel_len = _normalize_lengths(mel, mel_len)
        dmm_len = _normalize_lengths(dmm, dmm_len)
        au_len = _normalize_lengths(au, au_len)

        E_aud = self.mel_encoder(mel, mel_len)
        E_app = self.app_encoder(dmm, dmm_len)
        E_emo = self.emo_encoder(au, au_len)

        durations = torch.stack((
            mel_len.float() / self.mel_frame_hz,
            dmm_len.float() / self.video_frame_hz,
            au_len.float() / self.video_frame_hz,
        ), dim=1).amin(dim=1)
        target_lengths = torch.floor(durations * self.target_frame_hz).long().clamp(min=1)
        c, c_mask, aligned = self.fusion_mlp(
            E_aud, E_app, E_emo,
            mel_len, dmm_len, au_len, target_lengths,
        )

        if return_branches:
            result = (c, *aligned)
            return (*result, c_mask) if return_mask else result
        if return_mask:
            return c, c_mask
        return c

    # ──────────────────────────────────────────────────────────────────────
    # 파라미터 수 / 학습가능 파라미터 수 출력 유틸
    # ──────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """각 서브모듈의 파라미터 수를 딕셔너리로 반환."""
        def _count(module):
            total     = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable}

        result = {
            "mel_encoder" : _count(self.mel_encoder),
            "mfcc_ref_enc": _count(self.mfcc_ref_enc),
            "app_encoder" : _count(self.app_encoder),
            "emo_encoder" : _count(self.emo_encoder),
            "fusion_mlp"  : _count(self.fusion_mlp),
        }
        total_all     = sum(v["total"]     for v in result.values())
        trainable_all = sum(v["trainable"] for v in result.values())
        result["TOTAL"] = {"total": total_all, "trainable": trainable_all}
        return result

    def log_parameter_count(self):
        counts = self.count_parameters()
        logger.info("[SBE] 파라미터 수:")
        for name, cnt in counts.items():
            t, tr = cnt["total"], cnt["trainable"]
            logger.info(f"  {name:15s}: total={t:>10,}  trainable={tr:>10,}")
