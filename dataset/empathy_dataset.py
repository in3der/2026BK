"""Canonical Person-A/Person-B dataset for the integrated baseline.

The pre-integration dataset variants split *expanded responses*. That lets
different responses for the same ``conv_id`` appear in train and evaluation.
This module splits conversation IDs first and only then expands responses.

[데이터 구조 — /mnt/HDD1/bk_dataset]
JSON: generated_text/train_final_with_reference_images.json
  - conv_id       : "hit:0_conv:1"  → prefix "hit_0_conv_1"
  - input_gender  : "female" / "male"
  - input_context : { "emotion": "neutral", "reference_img": "hispanic_woman_2_neutral.png" }
  - responses[i]  : { "style": "Affective Listening",
                      "predicted_emotion": "happy",
                      "output_gender": "male",
                      "reference_img": "asian_man_2_happy.png" }

[파일 naming 규칙]
  input mel   : {prefix}_{female/male}_{Emotion}.pt       ← female/male
  input mfcc  : {prefix}_{female/male}_{Emotion}.npy      ← 동일
  input 3dmm  : {prefix}_{ref_img_stem}/  (폴더)          ← woman/man (reference_img 기반)
  input AU    : {prefix}_{ref_img_stem}.npy
  input exp   : {prefix}_{ref_img_stem}.npy
  output mel  : {prefix}_{style_clean}_{male/female}_{Emotion}.pt
  output 3dmm : {prefix}_{ref_img_stem}_{style_clean}/

[Gender 표기 혼재]
  JSON gender : "female" / "male"
  3dmm/AU/exp : "woman" / "man"  (reference_img 안에 들어있는 표기)
  mel/mfcc    : "female" / "male"
  → 3dmm/AU/exp 경로는 reference_img stem을 그대로 사용하면 됨 (변환 불필요)

[Emotion 정규화]
  JSON "surprised" → 파일명 "Surprise"  (95.1% 매칭 → 이 처리로 해결)

[반환 텐서 (per sample)]
  mel_in    : [T_mel,  80]   Speaker A mel-spectrogram
  dmm_in    : [T_dmm, 486]   Speaker A 3DMM (per-frame stack)
  au_in     : [T_au,   25]   Speaker A Action Units
  mfcc_in   : [T_mfcc, 78]   Speaker A MFCC (Stage-1 alignment teacher)
  mel_gt    : [T_out,  80]   Empathizer B GT mel-spectrogram
  style_lbl : int (0~5)      공감 style 레이블
  emo_lbl   : int (0~N-1)    Empathizer B 감정 클래스 레이블

[Collate]
  가변 길이 → 배치 내 max T 로 zero-padding + 길이 반환
"""

import os
import json
import logging
import random
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
SPLIT_SCHEMA_VERSION = 1


def _conversation_ids(raw_data: list) -> List[str]:
    """Return sorted unique conversation IDs and reject malformed duplicates."""
    conv_ids = [entry["conv_id"] for entry in raw_data]
    if len(conv_ids) != len(set(conv_ids)):
        raise ValueError("JSON contains duplicate conv_id entries; group split is ambiguous")
    return sorted(conv_ids)


def _source_fingerprint(conv_ids: List[str]) -> str:
    payload = "\n".join(sorted(conv_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_split_manifest(raw_data: list, val_ratio: float = 0.1,
                         test_ratio: float = 0.05, seed: int = 42) -> dict:
    """Build a deterministic, leakage-free conv_id group split manifest."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1")

    conv_ids = _conversation_ids(raw_data)
    shuffled = conv_ids.copy()
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    test_ids = sorted(shuffled[:n_test])
    val_ids = sorted(shuffled[n_test:n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val:])

    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "source_conv_count": n_total,
        "source_fingerprint_sha256": _source_fingerprint(conv_ids),
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }
    validate_split_manifest(manifest, raw_data)
    return manifest


def validate_split_manifest(manifest: dict, raw_data: Optional[list] = None) -> None:
    """Fail closed if any conversation appears in more than one split."""
    splits = manifest.get("splits", {})
    missing = [name for name in SPLIT_NAMES if name not in splits]
    if missing:
        raise ValueError(f"split manifest is missing: {missing}")

    sets = {name: set(splits[name]) for name in SPLIT_NAMES}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            example = sorted(overlap)[:5]
            raise ValueError(f"conv_id leakage between {left}/{right}: {example}")

    if raw_data is not None:
        source_ids = set(_conversation_ids(raw_data))
        assigned = sets["train"] | sets["val"] | sets["test"]
        if assigned != source_ids:
            missing_ids = sorted(source_ids - assigned)[:5]
            extra_ids = sorted(assigned - source_ids)[:5]
            raise ValueError(
                f"split manifest/source mismatch; missing={missing_ids}, extra={extra_ids}"
            )
        expected = manifest.get("source_fingerprint_sha256")
        actual = _source_fingerprint(sorted(source_ids))
        if expected != actual:
            raise ValueError("split manifest fingerprint does not match dataset JSON")


def load_or_create_split_manifest(raw_data: list, manifest_path: Optional[str],
                                  val_ratio: float = 0.1,
                                  test_ratio: float = 0.05,
                                  seed: int = 42) -> dict:
    """Load a fixed manifest, or create it atomically when a path is supplied."""
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        validate_split_manifest(manifest, raw_data)
        return manifest

    manifest = build_split_manifest(raw_data, val_ratio, test_ratio, seed)
    if manifest_path:
        parent = os.path.dirname(os.path.abspath(manifest_path))
        os.makedirs(parent, exist_ok=True)
        tmp_path = f"{manifest_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, manifest_path)
        logger.info("[Dataset] split manifest created: %s", manifest_path)
    return manifest

# ──────────────────────────────────────────────────────────────────────────────
# 레이블 매핑 상수
# ──────────────────────────────────────────────────────────────────────────────

# 6가지 공감 style (JSON style 값 → 인덱스)
STYLE_LIST = [
    "Affective Listening",    # 0
    "Cognitive Empathy",      # 1
    "Humor/Lighthearted",     # 2
    "Practical Advice",       # 3
    "Reflective/Mirroring",   # 4
    "Supportive/Encouraging", # 5
]
STYLE2IDX: Dict[str, int] = {s: i for i, s in enumerate(STYLE_LIST)}

# style → 파일명 변환 ("/" → "_", 공백 → "_")
def style_to_filename(style: str) -> str:
    return style.replace("/", "_").replace(" ", "_")

# Emotion 정규화 (JSON predicted_emotion → 파일명 emotion 문자열)
# 핵심: "surprised" → "Surprise" (파일명 기준)
EMOTION_FILE_MAP: Dict[str, str] = {
    "neutral":   "Neutral",
    "happy":     "Happy",
    "sad":       "Sad",
    "surprise":  "Surprise",
    "surprised": "Surprise",   # ← 여기서 95.1% → ~100% 매칭
    "angry":     "Angry",
    "anger":     "Angry",
    "fear":      "Fear",
    "fearful":   "Fear",
    "disgust":   "Disgust",
    "disgusted": "Disgust",
}

# 감정 클래스 인덱스 (정규화된 소문자 기준)
EMOTION_CLASSES = sorted({"neutral", "happy", "sad", "surprise",
                           "angry", "fear", "disgust"})
EMOTION2IDX: Dict[str, int] = {e: i for i, e in enumerate(EMOTION_CLASSES)}

def normalize_emotion(raw: str) -> str:
    """JSON emotion 값을 정규화된 클래스 이름으로 변환."""
    raw = raw.lower().strip()
    mapping = {
        "surprised": "surprise",
        "anger":     "angry",
        "fearful":   "fear",
        "disgusted": "disgust",
    }
    return mapping.get(raw, raw)


# ──────────────────────────────────────────────────────────────────────────────
# 파일 경로 조립 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def build_input_paths(data_dir: str, prefix: str,
                      input_gender: str, input_emotion_raw: str,
                      input_ref_stem: str) -> Dict[str, str]:
    """
    Speaker A(입력) 의 각 feature 파일 경로를 조립한다.

    Args:
        data_dir        : dataset 루트 폴더
        prefix          : conv_id 변환 결과 ("hit_0_conv_1")
        input_gender    : "female" / "male"
        input_emotion_raw: JSON input_context.emotion ("neutral" 등)
        input_ref_stem  : reference_img 에서 .png 제거한 stem
                          ("hispanic_woman_2_neutral" 등)
    """
    emotion_cap = EMOTION_FILE_MAP.get(input_emotion_raw.lower(), input_emotion_raw.capitalize())

    return {
        # mel: female/male + Emotion (대문자 첫글자)
        "mel":  os.path.join(data_dir, "generated_input_audio_mel_official",
                             f"{prefix}_{input_gender}_{emotion_cap}.pt"),
        # mfcc: 동일 패턴
        "mfcc": os.path.join(data_dir, "generated_input_audio_mfcc",
                             f"{prefix}_{input_gender}_{emotion_cap}.npy"),
        # 3dmm: ref_img stem 기반 폴더
        "dmm":  os.path.join(data_dir, "generated_input_video_3dmm",
                             f"{prefix}_{input_ref_stem}"),
        # AU: ref_img stem 기반
        "au":   os.path.join(data_dir, "generated_input_video_AU",
                             f"{prefix}_{input_ref_stem}.npy"),
    }


def build_output_paths(data_dir: str, prefix: str,
                       style: str, output_gender: str,
                       output_emotion_raw: str,
                       output_ref_stem: str) -> Dict[str, str]:
    """
    Empathizer B(출력) 의 각 feature 파일 경로를 조립한다.

    Args:
        style             : "Affective Listening" 등 JSON style 값
        output_gender     : "female" / "male"
        output_emotion_raw: JSON predicted_emotion ("happy" / "surprised" 등)
        output_ref_stem   : reference_img stem ("asian_man_2_happy" 등)
    """
    style_fn    = style_to_filename(style)
    emotion_cap = EMOTION_FILE_MAP.get(output_emotion_raw.lower(),
                                       output_emotion_raw.capitalize())

    return {
        # mel: style + gender + emotion
        "mel_gt": os.path.join(data_dir, "generated_output_audio_mel_official",
                               f"{prefix}_{style_fn}_{output_gender}_{emotion_cap}.pt"),
        # 3dmm: ref_img_stem + style
        "dmm_gt": os.path.join(data_dir, "generated_output_video_3dmm",
                               f"{prefix}_{output_ref_stem}_{style_fn}"),
    }


def load_dmm_frames(dmm_dir: str) -> Optional[Tensor]:
    """
    3DMM 폴더 안의 per-frame npy 파일들을 [T, 486] 텐서로 스택.
    비어있거나 0-byte 파일은 skip.
    """
    if not os.path.isdir(dmm_dir):
        return None

    frame_files = sorted(
        f for f in os.listdir(dmm_dir)
        if f.endswith(".npy") and os.path.getsize(os.path.join(dmm_dir, f)) > 0
    )
    if not frame_files:
        return None

    frames = []
    for fn in frame_files:
        arr = np.load(os.path.join(dmm_dir, fn))  # (1, 486) 또는 (486,)
        frames.append(arr.squeeze(0) if arr.ndim == 2 else arr)

    return torch.tensor(np.stack(frames, axis=0), dtype=torch.float32)  # [T, 486]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EmpathyDataset(Dataset):
    """
    JSON 기반 speaker-empathizer 매칭 Dataset.

    각 sample = (speaker A 의 한 발화, empathizer B 의 한 공감 응답).
    JSON 의 한 entry 에 6개 responses 가 있으므로 총 sample 수 ≈ entries × 6.

    Parameters
    ----------
    json_path  : train_final_with_reference_images.json 경로
    data_dir   : /mnt/HDD1/bk_dataset
    split      : 'train', 'val', or 'test'
    val_ratio  : val split 비율 (기본 0.1)
    seed       : split 랜덤 시드
    verbose    : True 이면 매칭 실패 케이스 경고 출력
    """

    def __init__(self, json_path: str, data_dir: str,
                 split: str = "train", val_ratio: float = 0.1,
                 test_ratio: float = 0.05, seed: int = 42,
                 split_manifest_path: Optional[str] = None,
                 verbose: bool = False):
        super().__init__()
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}: {split}")
        self.data_dir = data_dir
        self.split    = split
        self.verbose  = verbose

        # JSON 로드
        logger.info(f"[Dataset] JSON 로드: {json_path}")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        logger.info(f"[Dataset] JSON entries: {len(raw_data)}")

        # conv_id를 먼저 분리한 뒤 response를 펼친다. 이 순서가 leakage를 막는다.
        self.split_manifest = load_or_create_split_manifest(
            raw_data=raw_data,
            manifest_path=split_manifest_path,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        selected_ids = set(self.split_manifest["splits"][split])
        selected_raw = [entry for entry in raw_data if entry["conv_id"] in selected_ids]

        all_samples = self._build_samples(selected_raw)
        logger.info(f"[Dataset] 전체 valid samples: {len(all_samples)} "
                    f"(selected conversations × 6 = {len(selected_raw)*6}, "
                    f"매칭 실패 제외)")
        self.samples = all_samples
        logger.info(
            "[Dataset] split=%s conversations=%d samples=%d",
            split, len(selected_ids), len(self.samples),
        )

        # 레이블 분포 요약
        self._log_label_distribution()

    def _build_samples(self, raw_data: list) -> list:
        """
        JSON 전체를 순회하며 valid sample (모든 파일이 존재하는 것) 만 수집.
        각 sample = dict of paths + labels.
        """
        samples = []
        miss_cnt = 0

        for entry in raw_data:
            # conv_id → prefix ("hit:0_conv:1" → "hit_0_conv_1")
            prefix     = entry["conv_id"].replace(":", "_")
            in_gender  = entry["input_gender"]           # "female" / "male"
            in_emo_raw = entry["input_context"]["emotion"]
            in_ref_img = entry["input_context"]["reference_img"]  # "hispanic_woman_2_neutral.png"
            in_ref_stem = in_ref_img.replace(".png", "")          # "hispanic_woman_2_neutral"

            # 입력 경로
            input_paths = build_input_paths(
                self.data_dir, prefix, in_gender, in_emo_raw, in_ref_stem
            )

            # 입력 파일 존재 확인
            input_ok = (
                os.path.isfile(input_paths["mel"]) and
                os.path.isfile(input_paths["mfcc"]) and
                os.path.isdir(input_paths["dmm"]) and
                os.path.isfile(input_paths["au"])
            )
            if not input_ok:
                if self.verbose:
                    logger.warning(f"[Dataset] 입력 파일 누락: prefix={prefix}")
                miss_cnt += 1
                continue

            for resp in entry["responses"]:
                style        = resp["style"]                # "Affective Listening" 등
                out_gender   = resp["output_gender"]        # "female" / "male"
                out_emo_raw  = resp["predicted_emotion"]    # "happy" / "surprised" 등
                out_ref_img  = resp["reference_img"]        # "asian_man_2_happy.png"
                out_ref_stem = out_ref_img.replace(".png", "")

                # 출력 경로
                output_paths = build_output_paths(
                    self.data_dir, prefix, style,
                    out_gender, out_emo_raw, out_ref_stem
                )

                # 출력 파일 존재 확인
                if not os.path.isfile(output_paths["mel_gt"]):
                    if self.verbose:
                        logger.warning(f"[Dataset] 출력 mel 누락: {output_paths['mel_gt']}")
                    miss_cnt += 1
                    continue

                # 레이블
                style_lbl = STYLE2IDX[style]
                emo_cls   = normalize_emotion(out_emo_raw)
                emo_lbl   = EMOTION2IDX.get(emo_cls, 0)  # 미등록 → neutral(0 or nearest)

                samples.append({
                    "conv_id":    entry["conv_id"],
                    "prefix":     prefix,
                    "mel_path":   input_paths["mel"],
                    "mfcc_path":  input_paths["mfcc"],
                    "dmm_dir":    input_paths["dmm"],
                    "au_path":    input_paths["au"],
                    "mel_gt_path": output_paths["mel_gt"],
                    "style_lbl":  style_lbl,
                    "emo_lbl":    emo_lbl,
                    # 디버깅용 메타데이터
                    "style_name": style,
                    "emo_name":   emo_cls,
                    "input_gender": in_gender,
                    "input_emotion": in_emo_raw,
                    "output_gender": out_gender,
                    "output_emotion": out_emo_raw,
                })

        if miss_cnt > 0:
            logger.warning(f"[Dataset] 총 {miss_cnt} 건 누락/스킵됨")
        return samples

    def _log_label_distribution(self):
        """style / emotion 레이블 분포를 로그로 출력."""
        style_cnt = [0] * len(STYLE_LIST)
        emo_cnt   = [0] * len(EMOTION_CLASSES)
        for s in self.samples:
            style_cnt[s["style_lbl"]] += 1
            emo_cnt[s["emo_lbl"]] += 1

        logger.info(f"[Dataset][{self.split}] Style 분포:")
        for i, sname in enumerate(STYLE_LIST):
            logger.info(f"  [{i}] {sname:30s}: {style_cnt[i]:5d}")
        logger.info(f"[Dataset][{self.split}] Emotion 분포:")
        for i, ename in enumerate(EMOTION_CLASSES):
            logger.info(f"  [{i}] {ename:12s}: {emo_cnt[i]:5d}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # ── Speaker A (입력) ──────────────────────────────────────────────
        # mel [T_mel, 80]
        mel_in = torch.load(s["mel_path"], map_location="cpu")
        if mel_in.ndim == 1:
            mel_in = mel_in.unsqueeze(-1)  # 혹시 1D면 보정
        mel_in = mel_in.float()

        # mfcc [T_mfcc, 78]
        mfcc_in = torch.tensor(
            np.load(s["mfcc_path"]), dtype=torch.float32
        )

        # 3DMM [T_dmm, 486]
        dmm_in = load_dmm_frames(s["dmm_dir"])
        if dmm_in is None:
            # 로드 실패 시 dummy (학습 중 발생하면 로그)
            logger.warning(f"[Dataset] 3DMM 로드 실패, zero 대체: {s['dmm_dir']}")
            dmm_in = torch.zeros(1, 486)

        # AU [T_au, 25]
        au_in = torch.tensor(
            np.load(s["au_path"]), dtype=torch.float32
        )

        # ── Empathizer B (출력 GT) ─────────────────────────────────────────
        mel_gt = torch.load(s["mel_gt_path"], map_location="cpu").float()

        return {
            "mel_in":    mel_in,         # [T_mel,  80]
            "mfcc_in":   mfcc_in,        # [T_mfcc, 78]
            "dmm_in":    dmm_in,         # [T_dmm, 486]
            "au_in":     au_in,          # [T_au,   25]
            "mel_gt":    mel_gt,         # [T_out,  80]
            "style_lbl": torch.tensor(s["style_lbl"], dtype=torch.long),
            "emo_lbl":   torch.tensor(s["emo_lbl"],   dtype=torch.long),
            # 디버깅용
            "prefix":    s["prefix"],
            "style_name": s["style_name"],
            "emo_name":   s["emo_name"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate — 가변 길이 padding
# ──────────────────────────────────────────────────────────────────────────────

def _pad_sequence_batch(tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    [T_i, D] 텐서 리스트를 배치 max T 로 zero-padding.

    Returns:
        padded : [B, T_max, D]
        lengths: [B]  각 샘플의 원래 T
    """
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    T_max = lengths.max().item()
    D     = tensors[0].shape[1]
    B     = len(tensors)
    padded = torch.zeros(B, T_max, D, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0], :] = t
    return padded, lengths


def empathy_collate_fn(batch: list) -> dict:
    """
    EmpathyDataset 의 가변 길이 텐서를 padding 하여 배치로 만듦.

    Returns dict:
        mel_in    : [B, T_mel_max,  80]
        mel_in_len: [B]
        mfcc_in   : [B, T_mfcc_max, 78]
        mfcc_in_len: [B]
        dmm_in    : [B, T_dmm_max, 486]
        dmm_in_len: [B]
        au_in     : [B, T_au_max,   25]
        au_in_len : [B]
        mel_gt    : [B, T_gt_max,   80]
        mel_gt_len: [B]
        style_lbl : [B]
        emo_lbl   : [B]
        prefix    : List[str]   (디버깅용)
    """
    mel_in_pad,  mel_in_len  = _pad_sequence_batch([b["mel_in"]  for b in batch])
    mfcc_in_pad, mfcc_in_len = _pad_sequence_batch([b["mfcc_in"] for b in batch])
    dmm_in_pad,  dmm_in_len  = _pad_sequence_batch([b["dmm_in"]  for b in batch])
    au_in_pad,   au_in_len   = _pad_sequence_batch([b["au_in"]   for b in batch])
    mel_gt_pad,  mel_gt_len  = _pad_sequence_batch([b["mel_gt"]  for b in batch])

    return {
        "mel_in":     mel_in_pad,   "mel_in_len":  mel_in_len,
        "mfcc_in":    mfcc_in_pad,  "mfcc_in_len": mfcc_in_len,
        "dmm_in":     dmm_in_pad,   "dmm_in_len":  dmm_in_len,
        "au_in":      au_in_pad,    "au_in_len":   au_in_len,
        "mel_gt":     mel_gt_pad,   "mel_gt_len":  mel_gt_len,
        "style_lbl":  torch.stack([b["style_lbl"] for b in batch]),
        "emo_lbl":    torch.stack([b["emo_lbl"]   for b in batch]),
        "prefix":     [b["prefix"]    for b in batch],
        "style_name": [b["style_name"] for b in batch],
        "emo_name":   [b["emo_name"]   for b in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader 팩토리
# ──────────────────────────────────────────────────────────────────────────────

def get_empathy_dataloader(json_path: str, data_dir: str,
                           split: str = "train",
                           batch_size: int = 8,
                           num_workers: int = 4,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.05,
                           seed: int = 42,
                           split_manifest_path: Optional[str] = None,
                           verbose: bool = False) -> DataLoader:
    """
    EmpathyDataset + empathy_collate_fn 을 사용하는 DataLoader 반환.
    """
    dataset = EmpathyDataset(
        json_path=json_path, data_dir=data_dir,
        split=split, val_ratio=val_ratio,
        test_ratio=test_ratio, seed=seed,
        split_manifest_path=split_manifest_path,
        verbose=verbose,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=empathy_collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    logger.info(f"[DataLoader] split={split}  batch={batch_size}  "
                f"batches/epoch={len(loader)}  workers={num_workers}")
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# 단독 실행 시 간단 검증
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    DATA_DIR  = "/mnt/HDD1/bk_dataset"
    JSON_PATH = os.path.join(DATA_DIR, "generated_text",
                             "train_final_with_reference_images.json")

    logger.info("=== EmpathyDataset 단독 검증 ===")
    loader = get_empathy_dataloader(
        json_path=JSON_PATH, data_dir=DATA_DIR,
        split="train", batch_size=4, num_workers=0,
    )

    # 첫 배치 shape 확인
    batch = next(iter(loader))
    logger.info("--- 첫 배치 shape ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k:15s}: {v.shape}  dtype={v.dtype}")
        elif isinstance(v, list):
            logger.info(f"  {k:15s}: {v[:2]}")
    logger.info("단독 검증 완료 ✓")
