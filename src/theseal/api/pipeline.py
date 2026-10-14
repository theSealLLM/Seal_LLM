"""API가 쓰는 실제 기능 연결: 분류기 → 분류 후처리 → 프롬프트 → 답변 생성.

프롬프트: unknown은 label_unknown(pilsoo), 나머지는 프롬프트 담당 함수를 그대로 쓴다.
분류기는 (라벨 번호, 확률 배열)을 내고, router가 Classifier.labels로 이름을 붙인다. 그 뒤로는 라벨 이름을 쓴다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from theseal.classifier.classifier import Classifier
from theseal.classifier.config import DEFAULT_TASK, PROJECT_ROOT, FinalModelNotSetError, task_paths
from theseal.answer.answer_generator import generate, generate_stream
from theseal.common.config import ENV_FILE, LLMConfig, config_from_env
from theseal.prompt import build_label_specific_prompt, build_label_unknown_prompt
from theseal.common.schemas import AnswerResult, PromptSpec
from theseal.prompt import router
from theseal.prompt.router import RoutingResult


@dataclass
class ApiConfig:
    # final: config.py FINAL_MODEL · "<모델>/<실행 ID>": runs/classifier/<task>/<모델>/<실행 ID>/model ·
    # 그 밖에는 모델 폴더 경로 (저장소 기준 상대 경로 가능)
    api_classifier: str = "final"
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


def classifier_dir(config: ApiConfig) -> Path:
    paths = task_paths(config.api_classifier_task)
    name = config.api_classifier
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


def load_classifier(config: ApiConfig) -> Classifier:
    model_dir = classifier_dir(config)
    if not model_dir.exists():
        raise ClassifierNotReady(
            f"분류 모델이 없습니다: {model_dir} (train_*.py로 학습하거나 share_model.py pull로 받으세요)")
    return Classifier(model_dir, device=config.api_classifier_device)


def classifier_name(clf: Classifier) -> str:
    """<모델>/<실행 ID> (모델 폴더의 위 두 단계)"""
    return f"{clf.model_dir.parent.parent.name}/{clf.model_dir.parent.name}"


async def classify(clf: Classifier, text: str) -> RoutingResult:
    """① 분류기 → ② 경로 규칙 → ③ LLM 검증"""
    prediction = await run_in_threadpool(clf.predict, text)
    return await router.run(prediction, text, labels=clf.labels, model_dir=clf.model_dir)


def build_prompt(label: str, text: str) -> PromptSpec:
    if label == "unknown":
        return build_label_unknown_prompt(text)
    return build_label_specific_prompt(text, label)


async def answer(label: str, text: str, provider: str | None, llm_config: LLMConfig) -> AnswerResult:
    return await generate(build_prompt(label, text), provider, config=llm_config)


async def answer_stream(label: str, text: str, provider: str | None, llm_config: LLMConfig) -> AsyncIterator[dict]:
    """delta 이벤트 … → done 이벤트. unknown은 툴 유무와 상관없이 스트리밍하지 않고 done만 보낸다"""
    if label == "unknown":
        yield {"type": "done", "result": await answer(label, text, provider, llm_config)}
        return
    async for event in generate_stream(build_prompt(label, text), provider, config=llm_config):
        yield event
