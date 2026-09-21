import os
import sys
import json
import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.speaker_behavior_encoder import SpeakerBehaviorEncoder
from model.conditional_mel_decoder import SBEWithMelDecoder
from dataset.empathy_dataset import (
    build_input_paths, build_output_paths, load_dmm_frames,
    STYLE2IDX, EMOTION2IDX, EMOTION_FILE_MAP, normalize_emotion, style_to_filename,
    load_or_create_split_manifest,
)

import matplotlib.pyplot as plt

# GT랑 예측한 mel 파형 비교
def visualize_mel(gt_mel, pred_mel, save_dir):
    gt = gt_mel.detach().cpu().numpy()        # [T, 80]
    pred = pred_mel.detach().cpu().numpy()    # [T, 80]

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.title("GT Mel")
    plt.imshow(gt.T, aspect='auto', origin='lower')  # 🔥 transpose 중요
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.title("Predicted Mel")
    plt.imshow(pred.T, aspect='auto', origin='lower')
    plt.colorbar()

    plt.tight_layout()

    save_path = os.path.join(save_dir, "mel_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[Visualization] 저장 완료: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("infer")

DEFAULT_STAGE2_CKPT = os.environ.get(
    "PERFRDIFF_STAGE2_CKPT",
    "./checkpoints/sbe/stage2/checkpoint_stage2_best.pth",
)


# ──────────────────────────────────────────────────────────────────────────────
# Vocoder: NVIDIA HiFi-GAN (torch.hub)
# ──────────────────────────────────────────────────────────────────────────────

class MelToAudio:
    """mel [T, 80] → waveform. NVIDIA HiFi-GAN (torch.hub)."""

    def __init__(self, device: str = "cpu"):
        self.device    = device
        self._vocoder  = None

    def _load_vocoder(self):
        if self._vocoder is not None:
            return
        logger.info("[Vocoder] NVIDIA HiFi-GAN 로딩...")
        loaded = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan'
        )
        vocoder = loaded[0] if isinstance(loaded, tuple) else loaded
        self._vocoder = vocoder.to(self.device).eval()
        logger.info("[Vocoder] 로딩 완료 ✓")

    @torch.no_grad()
    def reconstruct(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [T, 80] → waveform [N_samples]."""
        self._load_vocoder()
        mel_in = mel.float().to(self.device)
        if mel_in.dim() == 2 and mel_in.shape[1] == 80:
            mel_in = mel_in.transpose(0, 1).unsqueeze(0)   # [1, 80, T]
        elif mel_in.dim() == 2 and mel_in.shape[0] == 80:
            mel_in = mel_in.unsqueeze(0)
        output = self._vocoder(mel_in)
        audio  = output[0] if isinstance(output, tuple) else output
        return audio.squeeze(0).cpu()

    def save_wav(self, mel: torch.Tensor, wav_path: str,
                 sample_rate: int = 22050):
        audio = self.reconstruct(mel)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(wav_path, audio, sample_rate)
        logger.info(f"[Vocoder] WAV 저장: {wav_path}  "
                    f"({audio.shape[-1] / sample_rate:.2f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# JSON 에서 유효한 샘플 수집
# ──────────────────────────────────────────────────────────────────────────────

def collect_valid_samples(json_path: str, data_dir: str,
                          split: str = "test",
                          split_manifest_path: str = "./artifacts/splits/conv_id_seed42.json",
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.05,
                          seed: int = 42,
                          max_scan: int = 0) -> list:
    with open(json_path) as f:
        raw_data = json.load(f)

    manifest = load_or_create_split_manifest(
        raw_data, split_manifest_path, val_ratio, test_ratio, seed
    )
    allowed_conv_ids = set(manifest["splits"][split])
    selected = [entry for entry in raw_data if entry["conv_id"] in allowed_conv_ids]
    if max_scan > 0:
        selected = selected[:max_scan]

    samples = []
    for entry in selected:
        prefix       = entry["conv_id"].replace(":", "_")
        in_gender    = entry["input_gender"]
        in_emo_raw   = entry["input_context"]["emotion"]
        in_ref_stem  = entry["input_context"]["reference_img"].replace(".png", "")

        input_paths = build_input_paths(
            data_dir, prefix, in_gender, in_emo_raw, in_ref_stem
        )
        if not (os.path.isfile(input_paths["mel"])  and
                os.path.isfile(input_paths["mfcc"]) and
                os.path.isdir(input_paths["dmm"])   and
                os.path.isfile(input_paths["au"])):
            continue

        for resp in entry["responses"]:
            style        = resp["style"]
            out_gender   = resp["output_gender"]
            out_emo_raw  = resp["predicted_emotion"]
            out_ref_stem = resp["reference_img"].replace(".png", "")

            output_paths = build_output_paths(
                data_dir, prefix, style, out_gender, out_emo_raw, out_ref_stem
            )
            if not os.path.isfile(output_paths["mel_gt"]):
                continue

            norm_emo = normalize_emotion(out_emo_raw)
            emo_idx  = EMOTION2IDX.get(norm_emo, 4)

            samples.append({
                "conv_id":        entry["conv_id"],
                "prefix":         prefix,
                "input_gender":   in_gender,
                "input_emotion":  in_emo_raw,
                "input_ref_stem": in_ref_stem,
                "style":          style,
                "style_idx":      STYLE2IDX.get(style, 0),
                "output_gender":  out_gender,
                "output_emotion": out_emo_raw,
                "output_emo_idx": emo_idx,
                "output_ref_stem": out_ref_stem,
                "mel_path":       input_paths["mel"],
                "mfcc_path":      input_paths["mfcc"],
                "dmm_dir":        input_paths["dmm"],
                "au_path":        input_paths["au"],
                "mel_gt_path":    output_paths["mel_gt"],
            })

    logger.info(
        "[Data] split=%s conversations=%d valid_samples=%d",
        split, len(allowed_conv_ids), len(samples),
    )
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# 단일 샘플 추론
# ──────────────────────────────────────────────────────────────────────────────

def infer_one_sample(model: SBEWithMelDecoder,
                     sample: dict,
                     vocoder: MelToAudio,
                     out_root: str,
                     device: str,
                     T_out: int):
    """
    하나의 샘플에 대해 DDIM 추론 → WAV 저장.

    T_out: 생성할 mel 프레임 수 (0이면 Person-A 입력 mel 길이를 proxy로 사용).
    """
    prefix   = sample["prefix"]
    style    = sample["style"]
    save_dir = os.path.join(out_root, f"{prefix}_{style_to_filename(style)}")
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"\n{'─'*60}")
    logger.info(f"[Infer] {prefix} / {style}")
    logger.info(f"  Speaker  : {sample['input_gender']}, {sample['input_emotion']}")
    logger.info(f"  Empathizer: {sample['output_gender']}, {sample['output_emotion']}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────
    speaker_mel = torch.load(sample["mel_path"], map_location="cpu").float()
    if speaker_mel.dim() == 1:
        speaker_mel = speaker_mel.unsqueeze(-1)

    speaker_mfcc = torch.tensor(
        np.load(sample["mfcc_path"]), dtype=torch.float32
    )
    speaker_dmm = load_dmm_frames(sample["dmm_dir"])
    if speaker_dmm is None:
        speaker_dmm = torch.zeros(1, 486)
    speaker_au = torch.tensor(
        np.load(sample["au_path"]), dtype=torch.float32
    )
    gt_mel = torch.load(sample["mel_gt_path"], map_location="cpu").float()

    logger.info(f"  speaker_mel: {speaker_mel.shape}")
    logger.info(f"  speaker_3dmm: {speaker_dmm.shape}")
    logger.info(f"  speaker_au: {speaker_au.shape}")
    logger.info(f"  gt_mel: {gt_mel.shape}")

    # 생성 길이 결정
    # 추론 길이는 GT 응답에서 가져오지 않는다. 길이 예측기가 아직 없는 현재
    # baseline에서는 명시한 T_out 또는 Person-A 입력 길이를 proxy로 사용한다.
    t_out = T_out if T_out > 0 else speaker_mel.shape[0]
    logger.info(f"  T_out: {t_out} frames")

    # ── DDIM 추론 ─────────────────────────────────────────────────────────
    model.eval()
    mel_b  = speaker_mel.unsqueeze(0).to(device)    # [1, T_mel, 80]
    dmm_b  = speaker_dmm.unsqueeze(0).to(device)    # [1, T_dmm, 486]
    au_b   = speaker_au.unsqueeze(0).to(device)     # [1, T_au,  25]
    style_b = torch.tensor([sample["style_idx"]], device=device)
    emo_b   = torch.tensor([sample["output_emo_idx"]], device=device)

    logger.info("  [Decoder] DDIM sampling (50 steps)...")
    with torch.no_grad():
        mel_pred, c, E_aud, E_app, E_emo = model.sample(
            mel_in=mel_b, dmm_in=dmm_b, au_in=au_b,
            style_lbl=style_b, emo_lbl=emo_b,
            T_out=t_out,
            mel_in_len=torch.tensor([mel_b.shape[1]], device=device),
            dmm_in_len=torch.tensor([dmm_b.shape[1]], device=device),
            au_in_len=torch.tensor([au_b.shape[1]], device=device),
        )

    mel_pred_cpu = mel_pred.squeeze(0).cpu()   # [T_out, 80]
    c_cpu        = c.squeeze(0).cpu()
    logger.info(f"  predicted_mel: {mel_pred_cpu.shape}")
    logger.info(f"  c norm: {c_cpu.norm().item():.2f}")

    # [추가] mel 시각화
    visualize_mel(
        gt_mel,
        mel_pred_cpu,
        save_dir=save_dir
    )

    # [추가] 분포 확인 (진짜 중요)
    print("\n=== MEL STATS ===")
    print("GT   min/max/mean:",
          gt_mel.min().item(),
          gt_mel.max().item(),
          gt_mel.mean().item())

    print("Pred min/max/mean:",
          mel_pred_cpu.min().item(),
          mel_pred_cpu.max().item(),
          mel_pred_cpu.mean().item())

    # 생성된 Mel 데이터를 NumPy 배열(.npy)로 저장
    # ───────────────────────────────────────────────────────────────────
    logger.info("  [NPY] Mel-spectrogram 텐서를 .npy 형식으로 저장...")
    np.save(os.path.join(save_dir, "predicted_mel.npy"), mel_pred_cpu.numpy())
    np.save(os.path.join(save_dir, "gt_mel.npy"), gt_mel.numpy())

    # speaker_mel은 shape이 [T, 80, 1]일 수 있으므로 2D로 맞춰서 저장
    speaker_mel_np = speaker_mel.squeeze(-1).numpy() if speaker_mel.dim() == 3 else speaker_mel.numpy()
    np.save(os.path.join(save_dir, "speaker_mel.npy"), speaker_mel_np)
    # ───────────────────────────────────────────────────────────────────

    # ── Vocoder 복호화 ────────────────────────────────────────────────────
    logger.info("  [Vocoder] Speaker 오디오 복원...")

    # [수정] speaker_mel이 3D 텐서일 경우를 대비해 확실하게 2D로 차원 축소 (unsqueeze 제거 효과)
    speaker_mel_2d = speaker_mel.squeeze(-1) if speaker_mel.dim() == 3 else speaker_mel
    vocoder.save_wav(speaker_mel_2d, os.path.join(save_dir, "speaker_audio.wav"))

    # [수정] 기존 중복 작성되었던 Speaker 오디오 복원 코드 블록 제거됨

    logger.info("  [Vocoder] 예측 mel(DDIM) 복호화...")
    vocoder.save_wav(mel_pred_cpu, os.path.join(save_dir, "predicted_audio.wav"))

    logger.info("  [Vocoder] GT mel 복호화...")
    vocoder.save_wav(gt_mel, os.path.join(save_dir, "gt_audio.wav"))

    # ── 메타데이터 ────────────────────────────────────────────────────────
    metadata = {
        "conv_id":            sample["conv_id"],
        "prefix":             prefix,
        "style":              style,
        "input_gender":       sample["input_gender"],
        "input_emotion":      sample["input_emotion"],
        "output_gender":      sample["output_gender"],
        "output_emotion":     sample["output_emotion"],
        "decoder":            "ConditionalMelDecoder (DDIM 50steps)",
        "T_out":              t_out,
        "speaker_mel_shape":  list(speaker_mel.shape),
        "speaker_3dmm_shape": list(speaker_dmm.shape),
        "speaker_au_shape":   list(speaker_au.shape),
        "predicted_mel_shape": list(mel_pred_cpu.shape),
        "gt_mel_shape":        list(gt_mel.shape),
        "c_norm":             float(c_cpu.norm().item()),
        "E_aud_norm":         float(E_aud.squeeze(0).norm().item()),
        "E_app_norm":         float(E_app.squeeze(0).norm().item()),
        "E_emo_norm":         float(E_emo.squeeze(0).norm().item()),
        "files": ["speaker_audio.wav", "predicted_audio.wav",
                  "speaker_mel.npy", "predicted_mel.npy", "gt_mel.npy",
                  "gt_audio.wav", "metadata.json"],
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"[Infer] ✓ 완료: {save_dir}")
    return save_dir


# ──────────────────────────────────────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Integrated SBE+MelDecoder Inference")
    p.add_argument("--data_dir",  default="/mnt/HDD1/bk_dataset")
    p.add_argument("--json_path", default=None)
    p.add_argument("--ckpt_dir",  default="/home/ivpl-navi/PerFRDiff/checkpoints",
                   help="PerFRDiff 사전학습 ckpt 루트 (TransformerDenoiser 등)")
    p.add_argument("--ckpt", default=DEFAULT_STAGE2_CKPT,
                   help="Stage-2 checkpoint (.pth); PERFRDIFF_STAGE2_CKPT로 기본값 설정 가능")
    p.add_argument("--out_dir",   default="./outputs/inference")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--split_manifest", default="./artifacts/splits/conv_id_seed42.json")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.05)
    p.add_argument("--n_samples",  type=int, default=1)
    p.add_argument("--sample_idx", type=int, default=None)
    p.add_argument("--T_out",      type=int, default=0,
                   help="생성 프레임 수 (0이면 Person-A 입력 mel 길이 사용; GT 미사용)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default=None)
    p.add_argument("--cfg", type=float, default=7.5,
                   help="CFG 스케일 (기본값: 7.5, 낮출수록 부드러워짐)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = args.device
    logger.info(f"[Main] device: {device}")

    if args.json_path is None:
        args.json_path = os.path.join(
            args.data_dir, "generated_text",
            "train_final_with_reference_images.json"
        )
    os.makedirs(args.out_dir, exist_ok=True)

    # ── SBE 빌드 ─────────────────────────────────────────────────────────
    logger.info("\n[Main] SBE 빌드...")
    prior_ckpt = os.path.join(
        args.ckpt_dir, "diffusion_model",
        "DiffusionPriorNetwork", "checkpoint.pth"
    )
    sbe = SpeakerBehaviorEncoder.from_pretrained(
        ckpt_dir        = args.ckpt_dir,
        prior_ckpt_path = prior_ckpt if os.path.exists(prior_ckpt) else None,
        device          = device,
    )
    sbe.set_stage("main")

    # ── SBEWithMelDecoder 빌드 ────────────────────────────────────────────
    logger.info(f"\n[Main] ConditionalMelDecoder 빌드 (CFG Scale: {args.cfg})...")
    # [수정] args.cfg(Classifier-Free Guidance)를 인자로 명시적 전달
    model = SBEWithMelDecoder(sbe=sbe, ckpt_dir=args.ckpt_dir, guidance_scale=args.cfg)

    # ── Stage-2 ckpt 로드 ─────────────────────────────────────────────────
    if args.ckpt and os.path.isfile(args.ckpt):
        logger.info(f"\n[Main] Stage-2 체크포인트 로드: {args.ckpt}")
        payload = torch.load(args.ckpt, map_location=device)
        sd = payload.get("state_dict", payload)
        result = model.load_state_dict(sd, strict=False)
        logger.info(
            f"  로드 결과: missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)}"
        )
        if result.missing_keys:
            logger.info(f"  missing[:5]: {result.missing_keys[:5]}")
        logger.info(f"  epoch={payload.get('epoch','?')}, "
                    f"loss={payload.get('loss','?')}")
    else:
        raise FileNotFoundError(
            "Stage-2 checkpoint가 필요합니다. --ckpt 또는 "
            f"PERFRDIFF_STAGE2_CKPT를 설정하세요: {args.ckpt}"
        )

    # [수정] 모델 전체를 device로 이동 (디바이스 불일치 에러 방지)
    model.to(device)
    model.eval()

    # ── Vocoder ──────────────────────────────────────────────────────────
    vocoder = MelToAudio(device=device)

    # ── 유효 샘플 수집 ───────────────────────────────────────────────────
    logger.info("\n[Main] 유효 샘플 수집 중...")
    samples = collect_valid_samples(
        args.json_path, args.data_dir,
        split=args.split,
        split_manifest_path=args.split_manifest,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if not samples:
        logger.error("[Main] 유효한 샘플이 없습니다.")
        sys.exit(1)

    if args.sample_idx is not None:
        chosen = [samples[min(args.sample_idx, len(samples) - 1)]]
    else:
        random.seed(args.seed)
        chosen = random.sample(samples, min(args.n_samples, len(samples)))

    logger.info(f"[Main] 선택된 샘플: {len(chosen)}개")

    # ── 추론 루프 ────────────────────────────────────────────────────────
    result_dirs = []
    for i, sample in enumerate(chosen):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Sample {i+1}/{len(chosen)}")
        logger.info(f"{'='*60}")
        save_dir = infer_one_sample(
            model    = model,
            sample   = sample,
            vocoder  = vocoder,
            out_root = args.out_dir,
            device   = device,
            T_out    = args.T_out,
        )
        result_dirs.append(save_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"  추론 완료 — {len(result_dirs)} 샘플")
    for d in result_dirs:
        logger.info(f"  → {d}")


if __name__ == "__main__":
    main()
