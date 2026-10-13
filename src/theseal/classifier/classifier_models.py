"""사전학습 분류 모델(BERT 계열) 파인튜닝.

Trainer 대신 PyTorch 학습 루프를 직접 쓴다 (accelerate 의존성 없이 동작).
에폭마다 validation macro-F1을 보고 가장 좋았던 에폭의 가중치를 남긴다.
NeoBERT처럼 저장소 코드를 실행해야 하는 모델은 trust_remote_code=True로 불러온다.
"""

from __future__ import annotations

import inspect
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from .config import META_FILENAME, BertConfig
from .data_loader import LabelSet, Split
from .metrics import compute_metrics, load_json, now_iso, save_json

KIND = "bert"


def _logits(output) -> torch.Tensor:
    """모델 출력에서 logits만 꺼낸다 (저장소 코드 모델이 튜플을 돌려주는 경우 포함)."""
    return output.logits if hasattr(output, "logits") else output[0]


def _forward_keys(model) -> set[str] | None:
    """model.forward가 받는 인자 이름. **kwargs를 받으면 None (거르지 않음)."""
    params = inspect.signature(model.forward).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params}


def resolve_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> str:
    """혼합 정밀도는 CUDA에서만 쓴다. auto면 bf16을 지원하는 GPU는 bf16, 아니면 fp16."""
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"알 수 없는 precision: {precision!r}")
    if device.type != "cuda":
        return "fp32"
    if precision == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return precision


class TransformerClassifier:
    kind = KIND

    def __init__(
        self,
        model,
        tokenizer,
        label_set: LabelSet,
        *,
        max_length: int,
        base_model: str,
        device: torch.device,
        eval_batch_size: int = 64,
        trust_remote_code: bool = False,
    ) -> None:
        self._device = device
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.label_set = label_set
        self.max_length = max_length
        self.base_model = base_model
        self.eval_batch_size = eval_batch_size
        self.trust_remote_code = trust_remote_code
        # 토크나이저가 만든 입력 중 모델이 받지 않는 것(예: token_type_ids)은 넘기지 않는다
        self._input_keys = _forward_keys(self.model)

    @property
    def device(self) -> str:
        return str(self._device)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        label_set: LabelSet,
        *,
        max_length: int,
        device: str | None = None,
        local_files_only: bool = False,
        eval_batch_size: int = 64,
        trust_remote_code: bool = False,
    ) -> TransformerClassifier:
        """HF Hub 이름(예: klue/roberta-base) 또는 로컬 폴더에서 인코더를 읽고 분류 헤드를 새로 붙인다."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=len(label_set),
                id2label=label_set.id2label,
                label2id=label_set.label2id,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except ImportError as e:
            raise ImportError(
                f"{model_name_or_path}를 불러오는 데 필요한 패키지가 없습니다: {e}\n"
                "저장소 코드를 실행하는 모델(NeoBERT 등)은 xformers 같은 추가 패키지를 요구할 수 "
                "있습니다. 현재 torch 버전에 맞는 패키지를 설치하세요 (맥에서는 설치가 안 될 수 있음)."
            ) from e
        return cls(
            model,
            tokenizer,
            label_set,
            max_length=max_length,
            base_model=str(model_name_or_path),
            device=resolve_device(device),
            eval_batch_size=eval_batch_size,
            trust_remote_code=trust_remote_code,
        )

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            k: v.to(self._device)
            for k, v in enc.items()
            if self._input_keys is None or k in self._input_keys
        }

    def token_lengths(self, texts: list[str]) -> list[int]:
        """자르기 전 토큰 길이 (max_length가 충분한지 확인용)."""
        return [len(ids) for ids in self.tokenizer(texts, truncation=False)["input_ids"]]

    @torch.inference_mode()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        chunks = []
        for i in range(0, len(texts), self.eval_batch_size):
            logits = _logits(self.model(**self.encode(texts[i : i + self.eval_batch_size])))
            chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        if not chunks:
            return np.zeros((0, len(self.label_set)), dtype=np.float32)
        return np.concatenate(chunks)

    def predict(self, texts: list[str]) -> list[str]:
        return self.label_set.decode(self.predict_proba(texts).argmax(axis=1))

    def describe(self) -> dict:
        return {
            "num_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "base_model": self.base_model,
            "max_length": self.max_length,
        }

    def save(
        self,
        model_dir: Path,
        *,
        name: str,
        task: str,
        run: str | None = None,
        config: BertConfig | None = None,
    ) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        if self.trust_remote_code:
            # 저장소 코드 모델은 model.py 같은 코드 파일이 있어야 다시 불러올 수 있다.
            # save_pretrained가 복사하지 않았고 원본이 로컬 폴더면 거기서 복사한다.
            base = Path(self.base_model)
            if base.is_dir() and not any(model_dir.glob("*.py")):
                for src in base.glob("*.py"):
                    shutil.copy2(src, model_dir / src.name)
        save_json({
            "kind": KIND,
            "name": name,
            "run": run,
            "task": task,
            "labels": list(self.label_set.labels),
            "base_model": self.base_model,
            "max_length": self.max_length,
            "trust_remote_code": self.trust_remote_code,
            "config": asdict(config) if config else None,
            "saved_at": now_iso(),
        }, model_dir / META_FILENAME)

    @classmethod
    def load(cls, model_dir: Path, device: str | None = None) -> TransformerClassifier:
        model_dir = Path(model_dir)
        meta = load_json(model_dir / META_FILENAME)
        if meta["kind"] != KIND:
            raise ValueError(f"{model_dir}는 사전학습 분류 모델이 아닙니다 (kind={meta['kind']}).")
        label_set = LabelSet(tuple(meta["labels"]))
        trust = bool(meta.get("trust_remote_code", False))
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=trust
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=trust
        )
        if model.config.num_labels != len(label_set):
            raise ValueError(
                f"모델 라벨 수({model.config.num_labels})와 메타 라벨 수({len(label_set)})가 다릅니다."
            )
        return cls(
            model,
            tokenizer,
            label_set,
            max_length=meta["max_length"],
            base_model=meta["base_model"],
            device=resolve_device(device),
            trust_remote_code=trust,
        )


@dataclass
class TrainResult:
    history: list[dict]
    best_epoch: int
    best_val_macro_f1: float
    train_time_sec: float
    peak_gpu_memory_mb: float | None
    precision: str
    stopped_early: bool


def train_classifier(
    clf: TransformerClassifier,
    train: Split,
    val: Split,
    config: BertConfig,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """파인튜닝 후 validation macro-F1이 가장 높았던 에폭의 가중치로 되돌린다.

    train_time_sec에는 에폭마다 하는 validation 평가 시간도 들어간다.
    """
    set_seed(config.seed)
    device, model, label_set = clf._device, clf.model, clf.label_set

    precision = resolve_precision(config.precision, device)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    y_train = torch.tensor(label_set.encode(train.labels))
    y_val = np.array(label_set.encode(val.labels))
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    optimizer = torch.optim.AdamW(_param_groups(model, config.weight_decay), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    rng = np.random.default_rng(config.seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log(f"[train] device={device} precision={precision} steps/epoch={steps_per_epoch} "
        f"total_steps={total_steps}")

    history: list[dict] = []
    best_f1, best_epoch, best_state, wait, stopped_early = -1.0, 0, None, 0, False
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        order = rng.permutation(len(train))
        for step in range(steps_per_epoch):
            idx = order[step * config.batch_size : (step + 1) * config.batch_size]
            enc = clf.encode([train.texts[i] for i in idx])
            labels = y_train[idx].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                # 손실은 직접 계산한다 (저장소 코드 모델이 labels를 받지 않거나 loss를 안 줘도 동작)
                logits = _logits(model(**enc))
                loss = nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[epoch {epoch} step {step + 1}] loss가 {loss.item()}라 학습을 멈춥니다. "
                    "가중치가 제대로 불러와지지 않았거나(NeoBERT는 버전에 따라 NaN 문제가 보고됨) "
                    "학습률이 너무 클 수 있습니다."
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * len(idx)

        probs = clf.predict_proba(val.texts)
        true_probs = np.clip(probs[np.arange(len(y_val)), y_val], 1e-12, None)
        metrics = compute_metrics(y_val.tolist(), probs.argmax(axis=1).tolist(), label_set)
        record = {
            "epoch": epoch,
            "train_loss": round(loss_sum / len(train), 4),
            "val_loss": round(float(-np.log(true_probs).mean()), 4),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": round(time.perf_counter() - start, 1),
        }
        history.append(record)
        log(f"[epoch {epoch:>2}/{config.epochs}] train_loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} val_acc={record['val_accuracy']:.4f} "
            f"val_macro_f1={record['val_macro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, wait = metrics["macro_f1"], epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            wait += 1
            if config.patience and wait >= config.patience:
                stopped_early = True
                log(f"[train] {config.patience}에폭 동안 개선이 없어 조기 종료")
                break

    train_time = time.perf_counter() - start
    peak_mb = (
        round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
        if device.type == "cuda" else None
    )
    model.load_state_dict(best_state)
    model.eval()
    log(f"[train] best_epoch={best_epoch} val_macro_f1={best_f1:.4f} time={train_time:.1f}s")

    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_macro_f1=best_f1,
        train_time_sec=round(train_time, 2),
        peak_gpu_memory_mb=peak_mb,
        precision=precision,
        stopped_early=stopped_early,
    )


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """bias와 LayerNorm 가중치(1차원 텐서)에는 weight decay를 주지 않는다."""
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
