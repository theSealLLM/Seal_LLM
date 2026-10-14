"""API가 쓰는 실제 기능 연결: 분류기 → 분류 후처리 → 프롬프트 → 답변 생성.

프롬프트: prompt/prompt_generator.py 한 곳에서 만든다. unknown은 범위 밖 프롬프트, 나머지는 공통 정책 프롬프트 B(기본값).
분류기는 ClassificationResult(text · label_num · label_name · scores)를 내고, router.run(result)가 최종 라벨을 정한다.
그 뒤로는 라벨 이름을 쓴다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from theseal.answer.answer_generator import generate, generate_stream
from theseal.classifier.classifier import Classifier
from theseal.classifier.config import DEFAULT_TASK, PROJECT_ROOT, FinalModelNotSetError, task_paths
from theseal.common.config import ENV_FILE, LLMConfig, config_from_env
from theseal.common.schemas import AnswerResult, PromptSpec
from theseal.prompt import build_common_policy_prompt, build_label_unknown_prompt, router
from theseal.prompt.router import RoutingResult


@dataclass
class ApiConfig:
    # 분류 모델은 classifier/config.py의 FINAL_MODEL[api_classifier_task]를 쓴다.
    # 모델 폴더(runs/classifier/<task>/<모델>/<실행 ID>/model)가 없으면 같은 설정으로 학습해 그 자리에 둔다 (README)
    api_classifier_task: str = DEFAULT_TASK
    api_classifier_device: str | None = None  # None이면 자동 (GPU가 있으면 cuda)
    # 브라우저에서 직접 호출을 허용할 프론트 주소 (쉼표로 구분). "*"면 전부, "off"면 CORS를 켜지 않는다.
    # 기본은 로컬 개발 서버 (Vite 5173 · CRA/Next 3000)
    api_cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
    )

    @classmethod
    def from_env(cls) -> ApiConfig:
        return config_from_env(cls, ENV_FILE)

    def cors_origins(self) -> list[str]:
        if self.api_cors_origins.strip().lower() == "off":
            return []
        return [o.strip().rstrip("/") for o in self.api_cors_origins.split(",") if o.strip()]


class ClassifierNotReady(Exception):
    """학습된 분류 모델을 찾지 못함 → config_error"""


def classifier_dir(name: str = "final", task: str = DEFAULT_TASK) -> Path:
    """분류 모델 폴더. API는 항상 final(FINAL_MODEL)을 쓴다.
    로컬 평가 스크립트는 "<모델>/<실행 ID>"나 모델 폴더 경로로 다른 모델도 고를 수 있다."""
    paths = task_paths(task)
    if name == "final":
        try:
            return paths.final_model_dir
        except FinalModelNotSetError as e:
            raise ClassifierNotReady(str(e)) from None
    model, sep, run = name.partition("/")
    if sep and model and run and "/" not in run and not Path(name).exists():
        return paths.model_dir(model, run)
    path = Path(name)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_classifier(config: ApiConfig, name: str = "final") -> Classifier:
    model_dir = classifier_dir(name, config.api_classifier_task)
    if not model_dir.exists():
        raise ClassifierNotReady(
            f"분류 모델이 없습니다: {model_dir} (FINAL_MODEL과 같은 설정으로 train_*.py를 돌려 이 폴더에 두거나 "
            "share_model.py pull로 받으세요)")
    return Classifier(model_dir, device=config.api_classifier_device)


def classifier_name(clf: Classifier) -> str:
    """<모델>/<실행 ID> (모델 폴더의 위 두 단계)"""
    return f"{clf.model_dir.parent.parent.name}/{clf.model_dir.parent.name}"


async def classify(clf: Classifier, text: str) -> RoutingResult:
    """① 분류기 → ② 경로 규칙 → ③ LLM 검증"""
    prediction = await run_in_threadpool(clf.predict, text)
    return await router.run(prediction)  # 원문은 prediction.text


def build_prompt(label: str, text: str) -> PromptSpec:
    """unknown은 범위 밖 프롬프트(툴 사용), 나머지는 공통 정책 프롬프트(B).
    B는 프롬프트 A/B 실험(runs/llm/prompt_comparison, PR #6)에서 라벨별 프롬프트(A)보다 나아 선정됐다."""
    if label == "unknown":
        return build_label_unknown_prompt(text)
    return build_common_policy_prompt(text, label)


async def answer(label: str, text: str, provider: str | None, llm_config: LLMConfig) -> AnswerResult:
    return await generate(build_prompt(label, text), provider, config=llm_config)


async def answer_stream(label: str, text: str, provider: str | None, llm_config: LLMConfig) -> AsyncIterator[dict]:
    """delta 이벤트 … → done 이벤트. unknown은 툴 유무와 상관없이 스트리밍하지 않고 done만 보낸다"""
    if label == "unknown":
        yield {"type": "done", "result": await answer(label, text, provider, llm_config)}
        return
    async for event in generate_stream(build_prompt(label, text), provider, config=llm_config):
        yield event
