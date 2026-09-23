import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.speaker_behavior_encoder import SpeakerBehaviorEncoder
from model.conditional_mel_decoder import SBEWithMelDecoder
from dataset.empathy_dataset import (
    get_empathy_dataloader, STYLE_LIST, EMOTION_CLASSES
)

# ──────────────────────────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("train")


# ──────────────────────────────────────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Integrated SBE Training Pipeline")

    # 경로
    p.add_argument("--data_dir",  default="/mnt/HDD1/bk_dataset")
    p.add_argument("--json_path", default="/mnt/HDD1/bk_dataset/generated_text/train_final_with_reference_images.json",
                   help="train JSON 경로 (미지정 시 data_dir 아래 자동 탐색)")
    p.add_argument("--ckpt_dir",  default="/home/ivpl-navi/PerFRDiff/checkpoints",
                   help="PerFRDiff 사전학습 ckpt 루트")
    p.add_argument("--out_dir",   default="./outputs/checkpoints/sbe",
                   help="학습 결과 저장 폴더")
    p.add_argument("--tb_dir",    default="./outputs/runs/sbe",
                   help="TensorBoard log 폴더")
    p.add_argument("--log_file",  default="./outputs/logs/train.log",
                   help="파일 로그 경로 (미지정 시 콘솔만)")

    # 학습 단계
    p.add_argument("--stage", default="all",
                   choices=["align", "main", "all"],
                   help="align: Stage-1만 | main: Stage-2만 | all: 1→2 순서대로")
    p.add_argument("--stage1_epochs", type=int, default=2,
                   help="Stage-1 (alignment) epoch 수")
    p.add_argument("--stage2_epochs", type=int, default=200,
                   help="Stage-2 (main) epoch 수")
    p.add_argument("--resume_stage2", default=None,
                   help="Stage-2 재개 시 Stage-1 완료 체크포인트 경로")
    p.add_argument("--resume_stage2_full", default=None,
                   help="Stage-2 체크포인트에서 전체 상태 복원 후 이어서 학습")

    # 학습 하이퍼파라미터
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr_stage1",  type=float, default=1e-3,
                   help="Stage-1 학습률 (mel_encoder 만 학습)")
    p.add_argument("--lr_stage2",  type=float, default=3e-5,
                   help="Stage-2 기본 학습률 (mel_encoder, emo_encoder, fusion_mlp)")
    p.add_argument("--lr_stage2_transformer", type=float, default=3e-6,
                   help="Stage-2 app_encoder(Transformer) 학습률 (불안정하므로 낮게)")
    p.add_argument("--warmup_steps", type=int, default=300,
                   help="Stage-2 초반 LR warmup steps (gradient 폭발 방지)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--test_ratio",  type=float, default=0.05)
    p.add_argument("--split_manifest", default="./artifacts/splits/conv_id_seed42.json")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--save_every",  type=int, default=5,
                   help="Stage-2: N epoch 마다 체크포인트 저장")
    p.add_argument("--log_every",   type=int, default=50,
                   help="N 배치마다 로그 출력")
    p.add_argument("--max_train_steps", type=int, default=0,
                   help="0이면 전체 epoch, 양수이면 smoke test용 batch 제한")
    p.add_argument("--max_val_steps", type=int, default=0,
                   help="0이면 전체 validation, 양수이면 smoke test용 batch 제한")

    # 모델
    p.add_argument("--device", default=None,
                   help="'cuda:0' 등 (미지정 시 자동 감지)")
    p.add_argument("--gpus", default=None,
                   help="멀티 GPU (예: '0,1,2,3'). 지정 시 DataParallel 사용.")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """배치 손실 이동 평균 추적."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def unwrap_model(model: nn.Module) -> nn.Module:
    """DataParallel 래핑된 모델에서 원본 모듈 추출."""
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(sbe: SpeakerBehaviorEncoder, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, path: str, extra: dict = None):
    """SBE 체크포인트 저장."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch":      epoch,
        "loss":       loss,
        "state_dict": sbe.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info(f"[CKPT] 저장: {path}  (epoch={epoch}, loss={loss:.6f})")


def load_stage1_checkpoint(sbe: SpeakerBehaviorEncoder,
                           ckpt_path: str, device: str) -> int:
    """Stage-1 체크포인트에서 mel_encoder 가중치 복원."""
    assert os.path.isfile(ckpt_path), f"Stage-1 체크포인트 없음: {ckpt_path}"
    payload = torch.load(ckpt_path, map_location=device)
    sd = payload["state_dict"]

    # mel_encoder 키만 추출하여 로드
    mel_sd = {k: v for k, v in sd.items() if k.startswith("mel_encoder.")}
    result = sbe.load_state_dict(mel_sd, strict=False)
    logger.info(f"[CKPT] Stage-1 mel_encoder 가중치 복원: {ckpt_path}")
    logger.info(f"  epoch={payload.get('epoch', '?')}  "
                f"loss={payload.get('loss', '?'):.6f}")
    logger.info(f"  복원 키: {len(mel_sd)} 개  "
                f"missing: {result.missing_keys[:3]}")
    return payload.get("epoch", 0)


# ──────────────────────────────────────────────────────────────────────────────
# Stage-1: Manifold Alignment (mel → MFCC 공간 정렬)
# ──────────────────────────────────────────────────────────────────────────────

def run_stage1(sbe: SpeakerBehaviorEncoder,
               train_loader, val_loader,
               args, writer: SummaryWriter) -> str:
    """
    Stage-1 학습 루프.

    목적:
        mel_encoder(Linear 80→512)의 출력이
        mfcc_ref_enc(Linear 78→512, frozen)의 출력과 일치하도록 학습.

    참고:
        mel 과 mfcc 는 같은 오디오에서 추출되지만 hop size 차이로
        프레임 수(T)가 약간 다를 수 있음 → min(T_mel, T_mfcc) 로 자름.

    Returns:
        stage1_ckpt_path: 저장된 Stage-1 체크포인트 경로
    """
    logger.info("=" * 65)
    logger.info("  Stage-1 시작: Mel → MFCC 공간 Manifold Alignment")
    logger.info(f"  epochs={args.stage1_epochs}  lr={args.lr_stage1}")
    logger.info("=" * 65)

    sbe.set_stage("align")
    sbe.train()

    # mel_encoder 파라미터만 optimizer 에 등록
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, sbe.parameters()),
        lr=args.lr_stage1
    )
    trainable_params = sum(p.numel() for p in sbe.parameters() if p.requires_grad)
    logger.info(f"[Stage-1] 학습 파라미터 수: {trainable_params:,}  "
                f"(mel_encoder 만)")

    device = next(sbe.parameters()).device
    best_val_loss = float("inf")
    global_step = 0
    stage1_ckpt_path = os.path.join(args.out_dir, "stage1", "checkpoint_stage1.pth")

    for epoch in range(1, args.stage1_epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        sbe.train()
        meter = AverageMeter()
        t0 = time.time()

        for step, batch in enumerate(tqdm(train_loader,
                                          desc=f"[Stage-1] Epoch {epoch}/{args.stage1_epochs}",
                                          leave=False)):
            if args.max_train_steps and step >= args.max_train_steps:
                break
            mel_in  = batch["mel_in"].to(device)    # [B, T_mel,  80]
            mfcc_in = batch["mfcc_in"].to(device)   # [B, T_mfcc, 78]
            mel_in_len = batch["mel_in_len"].to(device)
            mfcc_in_len = batch["mfcc_in_len"].to(device)

            optimizer.zero_grad()
            loss = sbe.alignment_loss(
                mel_in, mfcc_in, mel_len=mel_in_len, mfcc_len=mfcc_in_len
            )
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), mel_in.shape[0])
            global_step += 1

            if (step + 1) % args.log_every == 0:
                logger.info(
                    f"[Stage-1] ep{epoch} step{step+1:4d}  "
                    f"loss={loss.item():.6f}  avg={meter.avg:.6f}"
                )
                if writer:
                    writer.add_scalar("Stage1/train_loss_step", loss.item(), global_step)

        epoch_time = time.time() - t0
        logger.info(
            f"[Stage-1] Epoch {epoch} 완료 | "
            f"train_loss={meter.avg:.6f}  time={epoch_time:.1f}s"
        )
        if writer:
            writer.add_scalar("Stage1/train_loss_epoch", meter.avg, epoch)

        # ── Validation ───────────────────────────────────────────────────
        sbe.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  [Val]", leave=False):
                if args.max_val_steps and val_meter.count >= args.max_val_steps * args.batch_size:
                    break
                mel_in  = batch["mel_in"].to(device)
                mfcc_in = batch["mfcc_in"].to(device)
                mel_in_len = batch["mel_in_len"].to(device)
                mfcc_in_len = batch["mfcc_in_len"].to(device)
                loss = sbe.alignment_loss(
                    mel_in, mfcc_in, mel_len=mel_in_len, mfcc_len=mfcc_in_len
                )
                val_meter.update(loss.item(), mel_in.shape[0])

        logger.info(
            f"[Stage-1] Epoch {epoch} Val | "
            f"val_loss={val_meter.avg:.6f}"
            + ("  ← Best!" if val_meter.avg < best_val_loss else "")
        )
        if writer:
            writer.add_scalar("Stage1/val_loss_epoch", val_meter.avg, epoch)

        if val_meter.avg < best_val_loss:
            best_val_loss = val_meter.avg
            save_checkpoint(
                sbe, optimizer, epoch, best_val_loss,
                stage1_ckpt_path,
                extra={"stage": "align", "best_val_loss": best_val_loss}
            )

    logger.info(f"[Stage-1] 완료 ✓  best_val_loss={best_val_loss:.6f}")
    logger.info(f"[Stage-1] 체크포인트: {stage1_ckpt_path}")
    return stage1_ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# Stage-2: SBE + ConditionalMelDecoder (Diffusion 기반) 학습
# ──────────────────────────────────────────────────────────────────────────────
# [변경 사항 — SBEWithPlaceholderDecoder 제거]
# 이전: Linear 디코더(placeholder) → 동일 값 반복 출력 → 삐- 소리
# 현재: TransformerDenoiser(nfeats=80) + DecoderLatentDiffusion
#       = 실제 mel 분포를 학습하는 DDPM/DDIM 기반 디코더
#
# SBEWithMelDecoder는 canonical conditional_mel_decoder.py 에 정의됨.
# (canonical imports are defined at the top of this file)
# ──────────────────────────────────────────────────────────────────────────────


def _make_stage2_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """
    모듈별 학습률을 다르게 설정한 AdamW optimizer.

    param group 분류:
      app_encoder  (9M, Transformer): lr_stage2_transformer (3e-6)
                                       — Transformer 불안정성 완화
      그 외 전체   (SBE + Decoder)  : lr_stage2             (3e-5)

    [팀 참고]
    - Decoder(TransformerDenoiser)는 기본 LR 적용 (pretrained partial 로드)
    - app_encoder만 10배 낮은 LR: c_norm 폭발 방지
    """
    raw = unwrap_model(model)

    app_params = list(raw.sbe.app_encoder.parameters())
    app_ids    = set(id(p) for p in app_params)

    other_params = [p for p in raw.parameters()
                    if p.requires_grad and id(p) not in app_ids]

    param_groups = [
        {"params": other_params,
         "lr": args.lr_stage2,
         "name": "default"},
        {"params": [p for p in app_params if p.requires_grad],
         "lr": args.lr_stage2_transformer,
         "name": "app_encoder"},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    logger.info(
        f"[Stage-2] Optimizer param groups:\n"
        f"  default     : lr={args.lr_stage2:.1e}  "
        f"params={sum(p.numel() for p in other_params if p.requires_grad):,}\n"
        f"  app_encoder : lr={args.lr_stage2_transformer:.1e}  "
        f"params={sum(p.numel() for p in app_params if p.requires_grad):,}"
    )
    return optimizer


def _make_warmup_cosine_scheduler(optimizer, warmup_steps: int,
                                   total_steps: int):
    """
    Warmup (선형) + CosineAnnealing 복합 스케줄러.

    초반 warmup_steps 동안 LR을 0 → 기본값으로 선형 증가,
    이후 CosineAnnealing으로 감소. gradient 폭발 방지에 효과적.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_stage2(model: SBEWithMelDecoder,
               train_loader, val_loader,
               args, writer: SummaryWriter,
               start_epoch: int = 0,
               resume_payload: dict = None):
    """
    Stage-2 학습 루프.

    Loss: DDPM x_0 prediction MSE (ConditionalMelDecoder.compute_loss)
          padding 제외, per-sample [B] → .mean() 집계

    주요 특징:
      - SBE + TransformerDenoiser(nfeats=80) + DecoderLatentDiffusion
      - Classifier-Free Guidance (drop_prob=0.2 학습, guidance_scale=7.5 추론)
      - 모듈별 차등 LR (app_encoder: 10배 낮게)
      - Warmup + CosineAnnealing
      - NaN 감지 시 배치 skip
      - 전체 모델(SBE+Decoder) state_dict 저장 → 추론 시 완전 복원 가능
    """
    logger.info("=" * 65)
    logger.info("  Stage-2 시작: SBE + ConditionalMelDecoder (DDPM)")
    logger.info(f"  epochs={args.stage2_epochs}")
    logger.info(f"  lr_default={args.lr_stage2:.1e}  "
                f"lr_transformer={args.lr_stage2_transformer:.1e}")
    logger.info(f"  warmup_steps={args.warmup_steps}")
    logger.info("=" * 65)

    unwrap_model(model).sbe.set_stage("main")
    model.train()

    optimizer   = _make_stage2_optimizer(model, args)
    total_steps = len(train_loader) * args.stage2_epochs
    scheduler   = _make_warmup_cosine_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps
    )

    device        = next(model.parameters()).device
    best_val_loss = float("inf")
    global_step   = 0
    nan_skip_cnt  = 0

    if resume_payload:
        if "optimizer" in resume_payload:
            try:
                optimizer.load_state_dict(resume_payload["optimizer"])
                logger.info("[Main] Stage-2 optimizer 상태 복원 완료")
            except (ValueError, KeyError) as exc:
                logger.warning(f"[Main] optimizer 상태가 현재 구성과 달라 새로 시작: {exc}")
        if "scheduler" in resume_payload:
            try:
                scheduler.load_state_dict(resume_payload["scheduler"])
                logger.info("[Main] Stage-2 scheduler 상태 복원 완료")
            except (ValueError, KeyError) as exc:
                logger.warning(f"[Main] scheduler 상태가 현재 구성과 달라 새로 시작: {exc}")
        best_val_loss = float(resume_payload.get("best_val_loss",
                                                 resume_payload.get("loss", float("inf"))))
        global_step = int(resume_payload.get("global_step", 0))
        nan_skip_cnt = int(resume_payload.get("nan_skip_total", 0))

    for epoch in range(start_epoch + 1, start_epoch + args.stage2_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        meter = AverageMeter()
        t0 = time.time()

        for step, batch in enumerate(tqdm(train_loader,
                                          desc=f"[Stage-2] Epoch {epoch}",
                                          leave=False)):
            if args.max_train_steps and step >= args.max_train_steps:
                break
            mel_in     = batch["mel_in"].to(device)      # [B, T_mel,  80]
            dmm_in     = batch["dmm_in"].to(device)      # [B, T_dmm, 486]
            au_in      = batch["au_in"].to(device)       # [B, T_au,   25]
            mel_gt     = batch["mel_gt"].to(device)      # [B, T_gt,   80]
            mel_gt_len = batch["mel_gt_len"].to(device)  # [B]
            mel_in_len = batch["mel_in_len"].to(device)
            dmm_in_len = batch["dmm_in_len"].to(device)
            au_in_len  = batch["au_in_len"].to(device)
            style_lbl  = batch["style_lbl"].to(device)   # [B]
            emo_lbl    = batch["emo_lbl"].to(device)     # [B]

            optimizer.zero_grad()

            # SBEWithMelDecoder.forward() → (loss_per [B], c, E_aud, E_app, E_emo)
            # DataParallel: 각 GPU가 [B/n] loss → gather → [B]
            loss_per, c, E_aud, E_app, E_emo = model(
                mel_in, dmm_in, au_in, mel_gt, mel_gt_len, style_lbl, emo_lbl,
                mel_in_len=mel_in_len, dmm_in_len=dmm_in_len, au_in_len=au_in_len,
            )
            loss = loss_per.mean()   # DataParallel gather 후 집계

            # ── NaN 감지: 배치 skip ───────────────────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                nan_skip_cnt += 1
                logger.warning(
                    f"[Stage-2] ep{epoch} step{step+1}  NaN/Inf 감지 → skip "
                    f"(누적={nan_skip_cnt}, "
                    f"c_norm={c.norm(dim=-1).mean().item():.2f})"
                )
                optimizer.zero_grad()
                continue

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            meter.update(loss.item(), mel_in.shape[0])
            global_step += 1

            if (step + 1) % args.log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                c_norm     = c.norm(dim=-1).mean().item()
                logger.info(
                    f"[Stage-2] ep{epoch} step{step+1:4d}  "
                    f"loss={loss.item():.5f}  avg={meter.avg:.5f}  "
                    f"c_norm={c_norm:.2f}  "
                    f"grad={float(grad_norm):.3f}  lr={current_lr:.2e}"
                )
                if writer:
                    writer.add_scalar("Stage2/train_loss_step", loss.item(), global_step)
                    writer.add_scalar("Stage2/c_norm",    c_norm,    global_step)
                    writer.add_scalar("Stage2/grad_norm", float(grad_norm), global_step)
                    writer.add_scalar("Stage2/lr",        current_lr, global_step)
                    writer.add_scalar("Stage2/E_aud_norm", E_aud.norm(dim=-1).mean().item(), global_step)
                    writer.add_scalar("Stage2/E_app_norm", E_app.norm(dim=-1).mean().item(), global_step)
                    writer.add_scalar("Stage2/E_emo_norm", E_emo.norm(dim=-1).mean().item(), global_step)

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[Stage-2] Epoch {epoch} 완료 | "
            f"train_loss={meter.avg:.5f}  lr={current_lr:.2e}  "
            f"time={epoch_time:.1f}s  nan_skip={nan_skip_cnt}"
        )
        if writer:
            writer.add_scalar("Stage2/train_loss_epoch", meter.avg, epoch)
            writer.add_scalar("Stage2/nan_skip_cumul",   nan_skip_cnt, epoch)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for val_step, batch in enumerate(tqdm(val_loader, desc="  [Val]", leave=False)):
                if args.max_val_steps and val_step >= args.max_val_steps:
                    break
                mel_in     = batch["mel_in"].to(device)
                dmm_in     = batch["dmm_in"].to(device)
                au_in      = batch["au_in"].to(device)
                mel_gt     = batch["mel_gt"].to(device)
                mel_gt_len = batch["mel_gt_len"].to(device)
                mel_in_len = batch["mel_in_len"].to(device)
                dmm_in_len = batch["dmm_in_len"].to(device)
                au_in_len  = batch["au_in_len"].to(device)
                style_lbl  = batch["style_lbl"].to(device)
                emo_lbl    = batch["emo_lbl"].to(device)

                loss_per, c, _, _, _ = model(
                    mel_in, dmm_in, au_in, mel_gt, mel_gt_len, style_lbl, emo_lbl,
                    mel_in_len=mel_in_len, dmm_in_len=dmm_in_len, au_in_len=au_in_len,
                )
                loss = loss_per.mean()
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_meter.update(loss.item(), mel_in.shape[0])

        is_best = val_meter.avg < best_val_loss
        logger.info(
            f"[Stage-2] Epoch {epoch} Val | "
            f"val_loss={val_meter.avg:.5f}"
            + ("  ← Best!" if is_best else "")
        )
        if writer:
            writer.add_scalar("Stage2/val_loss_epoch", val_meter.avg, epoch)

        if is_best:
            best_val_loss = val_meter.avg

        # ── 체크포인트 저장 ──────────────────────────────────────────────
        # 전체 모델(SBE+Decoder) state_dict 저장 → infer.py에서 완전 복원
        if (epoch % args.save_every == 0) or is_best:
            tag  = "best" if is_best else f"ep{epoch:03d}"
            path = os.path.join(args.out_dir, "stage2", f"checkpoint_stage2_{tag}.pth")
            raw_model = unwrap_model(model)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "epoch":       epoch,
                "loss":        val_meter.avg,
                "state_dict":  raw_model.state_dict(),   # SBE + Decoder 전체
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "stage":       "main",
                "best_val_loss": best_val_loss,
                "global_step": global_step,
                "nan_skip_total": nan_skip_cnt,
            }
            torch.save(payload, path)
            logger.info(f"[CKPT] 저장: {path}  (epoch={epoch})")

    logger.info(f"[Stage-2] 완료 ✓  best_val_loss={best_val_loss:.5f}  "
                f"총 nan_skip={nan_skip_cnt}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 파일 로그 추가 ──────────────────────────────────────────────────
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s", "%H:%M:%S"
        ))
        logging.getLogger().addHandler(fh)

    # ── 디바이스 ────────────────────────────────────────────────────────
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        args.device = f"cuda:{gpu_ids[0]}"
    elif args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = args.device
    logger.info(f"[Main] device: {device}")
    logger.info(f"[Main] stage: {args.stage}")

    # ── JSON 경로 자동 탐색 ──────────────────────────────────────────────
    if args.json_path is None:
        args.json_path = os.path.join(
            args.data_dir, "generated_text",
            "train_final_with_reference_images.json"
        )
    logger.info(f"[Main] JSON: {args.json_path}")
    logger.info(f"[Main] data: {args.data_dir}")

    # ── DataLoader ──────────────────────────────────────────────────────
    logger.info("\n[Main] DataLoader 구성 중...")
    train_loader = get_empathy_dataloader(
        json_path=args.json_path, data_dir=args.data_dir,
        split="train", batch_size=args.batch_size,
        num_workers=args.num_workers, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, seed=args.seed,
        split_manifest_path=args.split_manifest,
    )
    val_loader = get_empathy_dataloader(
        json_path=args.json_path, data_dir=args.data_dir,
        split="val", batch_size=args.batch_size,
        num_workers=args.num_workers, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, seed=args.seed,
        split_manifest_path=args.split_manifest,
    )

    # ── SBE 빌드 + 사전학습 가중치 로드 ─────────────────────────────────
    logger.info("\n[Main] SBE 빌드 & 사전학습 가중치 로드...")
    prior_ckpt = os.path.join(
        args.ckpt_dir, "diffusion_model",
        "DiffusionPriorNetwork", "checkpoint.pth"
    )
    sbe = SpeakerBehaviorEncoder.from_pretrained(
        ckpt_dir       = args.ckpt_dir,
        prior_ckpt_path= prior_ckpt if os.path.exists(prior_ckpt) else None,
        device         = device,
    )
    sbe.log_parameter_count()

    # ── TensorBoard ─────────────────────────────────────────────────────
    os.makedirs(args.tb_dir, exist_ok=True)
    writer = SummaryWriter(args.tb_dir)
    logger.info(f"[Main] TensorBoard: {args.tb_dir}")

    stage1_ckpt = args.resume_stage2  # Stage-2 재개 시 이미 있을 경로

    # ── Stage-1 ─────────────────────────────────────────────────────────
    if args.stage in ("align", "all"):
        stage1_ckpt = run_stage1(sbe, train_loader, val_loader, args, writer)
        logger.info(f"\n[Main] Stage-1 완료 → 체크포인트: {stage1_ckpt}")

    # ── Stage-2 ─────────────────────────────────────────────────────────
    if args.stage in ("main", "all"):
        start_epoch = 0
        if args.resume_stage2_full:
            if not os.path.isfile(args.resume_stage2_full):
                raise FileNotFoundError(
                    f"Stage-2 resume checkpoint 없음: {args.resume_stage2_full}"
                )
            logger.info("[Main] Stage-2 전체 checkpoint를 우선 복원합니다")
        elif stage1_ckpt and os.path.isfile(stage1_ckpt):
            start_epoch = load_stage1_checkpoint(sbe, stage1_ckpt, device)
        else:
            logger.warning("[Main] Stage-1 체크포인트 없음 → 랜덤 초기화 상태로 진행")

        model = SBEWithMelDecoder(sbe=sbe, ckpt_dir=args.ckpt_dir).to(device)

        if args.gpus:
            gpu_ids = [int(g) for g in args.gpus.split(",")]
            model = nn.DataParallel(model, device_ids=gpu_ids)
            logger.info(f"[Main] DataParallel 활성: GPU {gpu_ids}")

        # ── Stage-2 체크포인트 전체 복원 (추가된 부분) ──────────────────────
        resume_payload = None
        if args.resume_stage2_full:
            logger.info(f"[Main] Stage-2 체크포인트 복원: {args.resume_stage2_full}")
            resume_payload = torch.load(args.resume_stage2_full, map_location=device)
            unwrap_model(model).load_state_dict(resume_payload["state_dict"])
            start_epoch = resume_payload["epoch"]
            logger.info(f"[Main] 복원 완료 → epoch={start_epoch}, "
                        f"best_val_loss={resume_payload.get('best_val_loss', resume_payload.get('loss', float('nan'))):.5f}")

        run_stage2(model, train_loader, val_loader, args, writer,
                   start_epoch=start_epoch, resume_payload=resume_payload)
        logger.info("[Main] Stage-2 완료 ✓")

    writer.close()
    logger.info("[Main] 학습 파이프라인 전체 완료 ✓")


if __name__ == "__main__":
    main()
