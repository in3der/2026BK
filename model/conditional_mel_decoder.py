import os
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.diffusion.diffusion_decoder.transformer_denoiser import TransformerDenoiser
from model.diffusion.gaussian_diffusion import DecoderLatentDiffusion
from model.diffusion.operator import position_encoding as _pe_module


# ──────────────────────────────────────────────────────────────────────────────
# Position Encoding 패치
# ──────────────────────────────────────────────────────────────────────────────
# PerFRDiff의 PositionEmbeddingSine1D.forward()는 x + pos 대신 pos만 반환하는
# 버그가 있고, PositionEmbeddingLearned1D는 max_len=500으로 고정되어 있음.
# 두 문제 모두 mel 시퀀스(최대 ~1035 프레임) + 조건 토큰(5+)에서 오류 발생.
#
# 해결: 길이 제한 없는 올바른 sinusoidal PE 클래스를 정의하고
# build_position_encoding이 반환하는 대신 denoiser 생성 후 직접 교체.
# ──────────────────────────────────────────────────────────────────────────────

class _UnboundedSinePE(nn.Module):
    """
    길이 제한 없는 sinusoidal position encoding.
    forward(x): x + PE(x.shape[0]) 를 반환.

    원본 PositionEmbeddingSine1D:
      - max_len=500 buffer (초과 시 IndexError)
      - forward가 pos만 반환 (x + pos 아님) → 시퀀스 대체 버그
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def _make_pe(self, length: int, device) -> torch.Tensor:
        """[length, 1, d_model] sinusoidal PE."""
        pe = torch.zeros(length, self.d_model, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:self.d_model // 2])
        return pe.unsqueeze(1)  # [length, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, B, d_model]  →  x + PE  (broadcast over B)."""
        pe = self._make_pe(x.shape[0], x.device)  # [T, 1, d_model]
        return x + pe  # broadcast: [T, B, d_model]


logger = logging.getLogger(__name__)


def _patch_position_encodings(denoiser: TransformerDenoiser) -> None:
    """
    TransformerDenoiser의 query_pos / mem_pos 를 _UnboundedSinePE로 교체.

    교체 이유:
      - PositionEmbeddingLearned1D: max_len=500, 초과 시 IndexError
      - PositionEmbeddingSine1D:    forward가 x+pos가 아닌 pos만 반환
        → xseq이 PE값으로 덮여 배치 dim이 1로 붕괴 → chunk(2) 실패
    """
    d = denoiser.latent_dim
    denoiser.query_pos = _UnboundedSinePE(d)
    denoiser.mem_pos   = _UnboundedSinePE(d)
    logger.info("[MelDecoder] query_pos / mem_pos → _UnboundedSinePE (길이 제한 없음)")


# ──────────────────────────────────────────────────────────────────────────────
# DecoderLatentDiffusion 설정 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

class _DiffusionCfg:
    """DecoderLatentDiffusion이 요구하는 cfg namespace."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _build_diffusion(noise_schedule: str = "cosine",
                     num_train_timesteps: int = 1000,
                     num_inference_timesteps: int = 50,
                     timestep_spacing: str = "linspace",
                     predict: str = "epsilon",
                     var_type: str = "fixed_large",
                     noise_std: float = 1.0,
                     k: int = 1) -> DecoderLatentDiffusion:
    cfg = _DiffusionCfg(
        noise_schedule     = noise_schedule,
        predict            = predict,
        var_type           = var_type,
        rescale_timesteps  = False,
        noise_std          = noise_std,
        k                  = k,
        timestep_spacing   = timestep_spacing,
    )
    return DecoderLatentDiffusion(cfg, num_train_timesteps, num_inference_timesteps)


# ──────────────────────────────────────────────────────────────────────────────
# ConditionalMelDecoder
# ──────────────────────────────────────────────────────────────────────────────

class ConditionalMelDecoder(nn.Module):
    """
    Conditional Mel-Spectrogram Diffusion Decoder.

    학습 (compute_loss):
        predict="epsilon": 노이즈 ε 예측 + Min-SNR-γ 가중치
        random t → q_sample(mel_gt, t) → denoiser(x_t, t, kwargs)
        → weighted(L1 + MSE + Focus)(prediction, ε) [valid frames only]

    추론 (sample):
        DDIM 50 steps + CFG (guidance_scale=7.5)
        → mel [B, T_out, 80]

    Conditioning tokens → TransformerDenoiser (token_concat 방식):
        speaker_latent_embed      : c      [B, 1, 512]  SBE fused
        speaker_audio_encodings   : E_aud  [B, 1, 512]  mel branch
        speaker_3dmm_encodings    : E_app  [B, 1, 512]  appearance branch
        speaker_emotion_encodings : E_emo  [B, 1, 512]  emotion branch
        listener_latent_embed     : LayerNorm(style_emb+emo_emb) [B, 1, 512]
        (listener_personal, past_listener_emotion: 사용 안 함 → drop_prob=1.0)
    """

    MEL_BINS     = 80
    LATENT_DIM   = 512
    NUM_STYLES   = 6
    NUM_EMOTIONS = 7

    # ── Mel 정규화 상수 ─────────────────────────────────────────────────
    # Log-mel spectrogram 대표 범위 (librosa default, sr=16k 기준)
    # 실제 데이터: min ≈ -11.5, max ≈ +2.0
    # 약간의 여유(margin)를 주어 [-12.0, +2.5] → [-1, 1] 매핑
    MEL_MIN = -12.0
    MEL_MAX =   2.5

    def __init__(self,
                 ckpt_dir: str              = "./checkpoints",
                 num_train_timesteps: int   = 1000,
                 num_inference_timesteps: int = 50,
                 noise_schedule: str        = "cosine",
                 predict: str               = "epsilon",  # "epsilon" or "start_x"
                 snr_gamma: float           = 5.0,         # Min-SNR-γ (Hang et al. 2023)
                 guidance_scale: float      = 7.5,
                 drop_prob: float           = 0.2,
                 num_layers: int            = 7,
                 num_heads: int             = 4,
                 ff_size: int               = 1024,
                 dropout: float             = 0.1,
                 ):
        super().__init__()
        self.snr_gamma = snr_gamma

        # ── Mel 정규화 상수 (register_buffer: GPU 이동 자동, state_dict 저장) ─
        self.register_buffer('mel_min', torch.tensor(self.MEL_MIN))
        self.register_buffer('mel_max', torch.tensor(self.MEL_MAX))

        # ── Empathizer Conditioning Embeddings ────────────────────────────
        self.style_embed = nn.Embedding(self.NUM_STYLES,   self.LATENT_DIM)
        self.emo_embed   = nn.Embedding(self.NUM_EMOTIONS, self.LATENT_DIM)
        self.cond_norm   = nn.LayerNorm(self.LATENT_DIM)
        nn.init.normal_(self.style_embed.weight, std=0.02)
        nn.init.normal_(self.emo_embed.weight,   std=0.02)

        # ── TransformerDenoiser (mel 80d 버전) ────────────────────────────
        #   s_audio/3dmm/emotion_dim=512: SBE가 이미 proj → Identity
        #   encode_emotion=True, encode_3dmm=True → Linear proj 없음
        self.denoiser = TransformerDenoiser(
            nfeats             = self.MEL_BINS,   # ← 25→80 핵심 변경
            latent_dim         = self.LATENT_DIM,
            ff_size            = ff_size,
            num_layers         = num_layers,
            num_heads          = num_heads,
            dropout            = dropout,
            activation         = "gelu",
            position_embedding = "learned", # 생성 후 _patch_position_encodings()로 교체됨
            arch               = "trans_enc",
            ablation_skip_connection = True,
            # SBE 출력이 이미 512d → encoder 내 proj 불필요
            encode_emotion     = True,            # E_emo [B,512] → Identity
            encode_3dmm        = True,            # E_app [B,512] → Identity
            s_audio_dim        = self.LATENT_DIM, # E_aud [B,512] → Identity
            s_emotion_dim      = self.LATENT_DIM, # (encode_emotion=True 시 미사용)
            s_3dmm_dim         = self.LATENT_DIM, # (encode_3dmm=True 시 미사용)
            l_embed_dim        = self.LATENT_DIM, # style+emo token [B,512]
            s_embed_dim        = self.LATENT_DIM, # c [B,512]
            personal_emb_dim   = self.LATENT_DIM,
            condition_concat   = "token_concat",
            concat             = "concat_first",
            guidance_scale     = guidance_scale,
            # Classifier-Free Guidance drop probs (학습 시 랜덤 drop)
            l_latent_embed_drop_prob   = drop_prob,  # style+emo 조건
            s_latent_embed_drop_prob   = drop_prob,  # c 조건
            s_audio_enc_drop_prob      = drop_prob,  # E_aud 조건
            s_3dmm_enc_drop_prob       = drop_prob,  # E_app 조건
            s_emotion_enc_drop_prob    = drop_prob,  # E_emo 조건
            l_personal_embed_drop_prob = 1.0,        # 미사용 → 항상 zeros
            past_l_emotion_drop_prob   = 1.0,        # 미사용 → 항상 zeros
        )

        # ── Pretrained TransformerDenoiser partial load ────────────────────
        self._load_denoiser_pretrained(ckpt_dir)

        # ── Position Encoding 교체 (length-unlimited sinusoidal) ───────────
        # PositionEmbeddingLearned1D(max_len=500)와 PositionEmbeddingSine1D의
        # 버그(x+pos 대신 pos만 반환)를 모두 우회
        _patch_position_encodings(self.denoiser)

        # ── Diffusion Scheduler (k=1) ──────────────────────────────────────
        self.diffusion = _build_diffusion(
            noise_schedule          = noise_schedule,
            num_train_timesteps     = num_train_timesteps,
            num_inference_timesteps = num_inference_timesteps,
            predict                 = predict,
            k                       = 1,
        )
        self.num_train_timesteps = num_train_timesteps

        # ── SNR 버퍼 (Min-SNR-γ loss weighting용) ─────────────────────────
        # numpy → float32 tensor, GPU 이동 자동 (register_buffer)
        ac = torch.from_numpy(self.diffusion.alphas_cumprod).float()
        self.register_buffer("alphas_cumprod_buf", ac)

        logger.info("[MelDecoder] ConditionalMelDecoder 초기화")
        logger.info(f"  denoiser  : nfeats=80, latent={self.LATENT_DIM}, "
                    f"layers={num_layers}, heads={num_heads}")
        logger.info(f"  diffusion : {noise_schedule} schedule, "
                    f"train_T={num_train_timesteps}, "
                    f"infer_T={num_inference_timesteps}, CFG={guidance_scale}")

    # ── 사전학습 가중치 partial load ─────────────────────────────────────────

    def _load_denoiser_pretrained(self, ckpt_dir: str):
        """
        TransformerDenoiser ckpt에서 shape-compatible 키만 복원.

        ✓ 호환 (복원):
            encoder.*, time_proj.*, time_embedding.*
            query_pos.*, mem_pos.*
            speaker_latent_proj.*, listener_latent_proj.*
            listener_personal_proj.*

        ✗ 불호환 (skip):
            to_emotion_embed.*  : Linear(25→512) vs Linear(80→512)
            to_emotion_feat.*   : Linear(512→25) vs Linear(512→80)
            speaker_audio_proj.*: Linear(78→512) vs Identity
            speaker_3dmm_proj.* : Linear(58→512) vs Identity (encode_3dmm=True)
            speaker_emotion_proj.*: Linear(25→512) vs Identity (encode_emotion=True)
        """
        ckpt_path = os.path.join(
            ckpt_dir, "diffusion_model", "TransformerDenoiser", "checkpoint.pth"
        )
        if not os.path.isfile(ckpt_path):
            logger.warning(
                f"[MelDecoder] TransformerDenoiser ckpt 없음: {ckpt_path} "
                f"→ 랜덤 초기화 (encoder/time_emb 등 사전학습 불가)"
            )
            return

        payload = torch.load(ckpt_path, map_location="cpu")
        ckpt_sd = payload.get("state_dict", payload)
        model_sd = self.denoiser.state_dict()

        loaded, skipped = [], []
        compatible_sd = {}
        for k, v in ckpt_sd.items():
            if k not in model_sd:
                skipped.append(f"[no key] {k}")
                continue
            if model_sd[k].shape != v.shape:
                skipped.append(
                    f"[shape] {k}: ckpt{tuple(v.shape)} "
                    f"vs model{tuple(model_sd[k].shape)}"
                )
                continue
            compatible_sd[k] = v
            loaded.append(k)

        self.denoiser.load_state_dict(compatible_sd, strict=False)
        logger.info(
            f"[MelDecoder][Denoiser] pretrained partial load: "
            f"{len(loaded)}/{len(ckpt_sd)} keys"
        )
        if skipped:
            for s in skipped[:6]:
                logger.info(f"  skip: {s}")
            if len(skipped) > 6:
                logger.info(f"  ... and {len(skipped)-6} more skipped")

    # ── Model kwargs 빌드 ────────────────────────────────────────────────────

    def build_model_kwargs(self, c, Eaud, Eapp, Eemo, style_lbl, emo_lbl):
        s_emb = self.style_embed(style_lbl)  # (B, 512)
        e_emb = self.emo_embed(emo_lbl)  # (B, 512)
        l_cond = self.cond_norm(s_emb + e_emb)  # (B, 512)

        # None으로 선언된 키는 제외하고 전달하여 CFG forward_with_cond_scale 에러 방지
        return {
            "speaker_latent_embed": c,
            "speaker_audio_encodings": Eaud,
            "speaker_3dmm_encodings": Eapp,
            "speaker_emotion_encodings": Eemo,
            "listener_latent_embed": l_cond.unsqueeze(1),
        }

    # ── Mel 정규화 / 역정규화 ──────────────────────────────────────────────────

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Log-mel [MEL_MIN, MEL_MAX] → [-1, 1]."""
        return (mel - self.mel_min) / (self.mel_max - self.mel_min) * 2.0 - 1.0

    def denormalize_mel(self, mel_norm: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → Log-mel [MEL_MIN, MEL_MAX]."""
        return (mel_norm + 1.0) / 2.0 * (self.mel_max - self.mel_min) + self.mel_min

    # ── 학습: DDPM Denoising Loss ─────────────────────────────────────────────

    def compute_loss(self,
                     mel_gt:     torch.Tensor,  # [B, T_max, 80] padded
                     mel_gt_len: torch.Tensor,  # [B]
                     c:          torch.Tensor,  # [B, T, 512]
                     E_aud:      torch.Tensor,
                     E_app:      torch.Tensor,
                     E_emo:      torch.Tensor,
                     style_lbl:  torch.Tensor,
                     emo_lbl:    torch.Tensor,
                     ) -> torch.Tensor:
        """
        Epsilon prediction loss with Min-SNR-γ weighting (Hang et al. 2023).

        학습 흐름:
            mel_gt → normalize [-1, 1]
            t ~ Uniform[0, T_train)
            x_t = √ᾱ_t · mel_norm + √(1-ᾱ_t) · ε
            pred = denoiser(x_t, t, model_kwargs)   ← predicts ε (epsilon mode)
            loss = min(SNR(t), γ)/SNR(t) · [L1 + 0.5·MSE + 2.0·Focus](pred, ε)

        [왜 Min-SNR-γ?]
            start_x + 균일 가중치: 고-t(고노이즈)에서 모델이 평균값 예측으로 수렴
                → flat mel (mean collapse)
            epsilon + 균일 가중치: 저-t(쉬운 denoise)에 과집중
                → 고-t step 미학습 → 노이즈 출력
            epsilon + Min-SNR-γ:
                weight(t) = min(SNR, γ) / SNR
                - 고-t (SNR ≪ γ): weight ≈ 1 → 전역 구조 학습 강제
                - 저-t 소-t (SNR ≫ γ): weight = γ/SNR ≪ 1 → 과집중 방지
                → 학습 내내 balanced, flat mel/노이즈 모두 방지

        [Energy Focus Loss]
            mel_norm (클린 mel) 기반 고에너지 프레임에 L1 페널티 가중.
            epsilon 모드: target=노이즈이므로 클린 mel 에너지로 mask 생성.

        Returns:
            loss_per_sample: [B]  (DataParallel gather 호환)
        """
        B, T_max, _ = mel_gt.shape
        device = mel_gt.device

        # ── Mel 정규화: [-12, 2.5] → [-1, 1] ──────────────────────────────
        mel_norm = self.normalize_mel(mel_gt)

        model_kwargs = self.build_model_kwargs(c, E_aud, E_app, E_emo, style_lbl, emo_lbl)


        # ── Uniform timestep sampling ───────────────────────────────────────
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        # ── Min-SNR-γ 가중치 계산 ──────────────────────────────────────────
        # SNR(t) = ᾱ_t / (1 - ᾱ_t)
        # weight  = min(SNR, γ) / SNR  ∈ (0, 1]
        #   고-t (SNR ≪ γ): weight → 1.0   (전역 구조, 평균 붕괴 방지)
        #   저-t (SNR ≫ γ): weight → γ/SNR (과집중 방지)
        ac_t = self.alphas_cumprod_buf[t]                             # [B]
        snr   = ac_t / (1.0 - ac_t).clamp(min=1e-6)                  # [B]
        snr_weight = torch.clamp(snr, max=self.snr_gamma) / snr.clamp(min=1e-8)  # [B]

        # ── Denoising (x_t 생성 + model forward) ──────────────────────────
        results = self.diffusion.denoise(
            self.denoiser, mel_norm, t, model_kwargs=model_kwargs
        )
        # k=1 → squeeze: [B, 1, T_max, 80] → [B, T_max, 80]
        pred   = results["prediction_emotion"].squeeze(1)  # predicted ε (epsilon mode)
        target = results["target_emotion"].squeeze(1)      # actual ε (epsilon mode)

        # ── Padding mask ───────────────────────────────────────────────────
        mask = (
            torch.arange(T_max, device=device).unsqueeze(0)
            < mel_gt_len.unsqueeze(1)
        )  # [B, T_max], True = valid frame
        valid_mask_3d = mask.unsqueeze(-1).float()                    # [B, T_max, 1]
        n_valid = mask.sum(dim=1).clamp(min=1).float() * self.MEL_BINS  # [B]

        # ── Energy Focus mask (클린 mel_norm 기반) ─────────────────────────
        # epsilon 모드: target = 노이즈이므로 클린 mel 에너지로 마스크 생성
        # (target 기반이면 가우시안 노이즈를 마스킹하는 것이 되어 무의미)
        energy_thresh = ((-7.0 - self.MEL_MIN) / (self.MEL_MAX - self.MEL_MIN)) * 2.0 - 1.0
        energy_mask = (mel_norm > energy_thresh).float() * valid_mask_3d  # [B, T_max, 1]

        # ── Per-sample loss (패딩 제외) ────────────────────────────────────
        l1_err    = (pred - target).abs()   * valid_mask_3d
        mse_err   = (pred - target).pow(2)  * valid_mask_3d
        focus_err = (pred - target).abs()   * energy_mask

        loss_l1    = l1_err.sum(dim=(1, 2))    / n_valid   # [B]
        loss_mse   = mse_err.sum(dim=(1, 2))   / n_valid   # [B]
        loss_focus = focus_err.sum(dim=(1, 2)) / n_valid   # [B]

        base_loss = loss_l1 + (0.5 * loss_mse) + (2.0 * loss_focus)  # [B]

        # ── Min-SNR-γ 가중치 적용 ──────────────────────────────────────────
        loss_per = snr_weight * base_loss  # [B], DataParallel gather 호환

        return loss_per

    # ── 추론: DDIM Sampling + CFG ─────────────────────────────────────────────

    @torch.no_grad()
    def sample(self,
               c:         torch.Tensor,  # [B, 512]
               E_aud:     torch.Tensor,
               E_app:     torch.Tensor,
               E_emo:     torch.Tensor,
               style_lbl: torch.Tensor,
               emo_lbl:   torch.Tensor,
               T_out:     int,
               ) -> torch.Tensor:
        """
        DDIM sampling (50 steps) + CFG (guidance_scale=7.5) → mel [B, T_out, 80].

        p_mean_variance → forward_with_cond_scale() 호출로 CFG 적용.
        T_out: 생성할 프레임 수 (보통 입력 mel과 동일 혹은 GT 길이 사용).
        """
        B = c.shape[0]
        model_kwargs = self.build_model_kwargs(c, E_aud, E_app, E_emo, style_lbl, emo_lbl)
        shape = (B, T_out, self.MEL_BINS)

        # ddim_sample_loop_progressive → p_mean_variance → forward_with_cond_scale
        outputs = list(self.diffusion.ddim_sample_loop_progressive(
            matcher       = None,  # 내부 루프에서 미사용
            model         = self.denoiser,
            model_kwargs  = model_kwargs,
            shape         = shape,
            progress      = False,
            eta           = 0.0,   # 결정론적 DDIM
            clip_denoised = True,  # ✅ 정규화된 [-1,1] 범위 → clip 유효
        ))
        # 마지막 step의 x_0 예측값 (정규화 상태 [-1, 1])
        mel_norm = outputs[-1]["decoded_prediction"]  # [B, T_out, 80]

        # ✅ 역정규화: [-1, 1] → 원본 Log-Mel 범위
        mel_pred = self.denormalize_mel(mel_norm)
        return mel_pred


# ──────────────────────────────────────────────────────────────────────────────
# SBEWithMelDecoder: canonical Stage-2 training/inference wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SBEWithMelDecoder(nn.Module):
    """
    Stage-2 통합 모듈: SpeakerBehaviorEncoder + ConditionalMelDecoder.

    [SBEWithPlaceholderDecoder 대체]
    이전: Linear 디코더 → 삐- 소리 (mel 다양성 없음)
    현재: TransformerDenoiser + DDPM/DDIM → 실제 mel 분포 학습

    학습 forward():
        → (loss_per_sample [B], c, E_aud, E_app, E_emo)
        DataParallel 호환: loss [B] 반환 → 호출자에서 .mean()

    추론 sample():
        → mel_pred [B, T_out, 80]
    """

    def __init__(self, sbe, ckpt_dir: str = "./checkpoints", **decoder_kwargs):
        super().__init__()
        self.sbe     = sbe
        self.decoder = ConditionalMelDecoder(ckpt_dir=ckpt_dir, **decoder_kwargs)

        logger.info("[Stage-2] SBEWithMelDecoder 초기화")
        logger.info("  SBE + ConditionalMelDecoder(TransformerDenoiser+DDPM)")

    def forward(self,
                mel_in:     torch.Tensor,  # [B, T_mel,  80]
                dmm_in:     torch.Tensor,  # [B, T_dmm, 486]
                au_in:      torch.Tensor,  # [B, T_au,   25]
                mel_gt:     torch.Tensor,  # [B, T_gt,   80]  패딩된 GT
                mel_gt_len: torch.Tensor,  # [B]
                style_lbl:  torch.Tensor,  # [B]
                emo_lbl:    torch.Tensor,  # [B]
                mel_in_len: torch.Tensor = None,
                dmm_in_len: torch.Tensor = None,
                au_in_len:  torch.Tensor = None,
                ):
        """
        Returns:
            loss_per_sample: [B]   — DataParallel gather 호환, .mean()은 호출자
            c:               [B, 512]
            E_aud, E_app, E_emo: [B, 512]
        """
        # SBE: 3 branch → fused c
        c, E_aud, E_app, E_emo = self.sbe(
            mel=mel_in, dmm=dmm_in, au=au_in,
            mel_len=mel_in_len, dmm_len=dmm_in_len, au_len=au_in_len,
            return_branches=True,
        )

        # Diffusion denoising loss [B]
        loss_per = self.decoder.compute_loss(
            mel_gt, mel_gt_len, c, E_aud, E_app, E_emo, style_lbl, emo_lbl
        )
        return loss_per, c, E_aud, E_app, E_emo

    @torch.no_grad()
    def sample(self,
               mel_in:    torch.Tensor,
               dmm_in:    torch.Tensor,
               au_in:     torch.Tensor,
               style_lbl: torch.Tensor,
               emo_lbl:   torch.Tensor,
               T_out:     int,
               mel_in_len: torch.Tensor = None,
               dmm_in_len: torch.Tensor = None,
               au_in_len:  torch.Tensor = None,
               ) -> tuple:
        """
        추론: DDIM sampling.

        Returns:
            mel_pred: [B, T_out, 80]
            c:        [B, 512]
        """
        c, E_aud, E_app, E_emo = self.sbe(
            mel=mel_in, dmm=dmm_in, au=au_in,
            mel_len=mel_in_len, dmm_len=dmm_in_len, au_len=au_in_len,
            return_branches=True,
        )
        mel_pred = self.decoder.sample(
            c, E_aud, E_app, E_emo, style_lbl, emo_lbl, T_out
        )
        return mel_pred, c, E_aud, E_app, E_emo
