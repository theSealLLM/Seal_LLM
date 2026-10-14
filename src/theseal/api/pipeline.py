"""API가 쓰는 실제 기능 연결 (LCEL 체인): 질문(text) → 분류기 → 분류 후처리 → 프롬프트 → 답변.

    answer_chain(clf).ainvoke({"text": ..., "label": None, "provider": None})  →  AnswerResponse

    {"text", "label"?, "provider"?}
      → classification  분류기(ClassificationResult) → router.run → RoutingResult (label을 주면 건너뜀)
      → label           최종 라벨 (주어진 label 또는 classification.label)
      → prompt          prompt_generator.prompt_for_label → PromptSpec
      → AnswerResponse  generate → AnswerResult + classification

단계마다 RunnablePassthrough.assign으로 결과를 쌓아서, 최종 응답에 분류 근거(classification)가 남는다.
스트리밍은 prompt까지 prepare_chain으로 만들고 답변만 answer_stream으로 흘린다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from theseal.answer.answer_generator import generate, generate_stream
from theseal.classifier.classifier import Classifier
from theseal.classifier.config import DEFAULT_TASK, PROJECT_ROOT, FinalModelNotSetError, task_paths
from theseal.common.config import ENV_FILE, LLMConfig, config_from_env, get_config
from theseal.common.schemas import PromptSpec
from theseal.prompt import router
from theseal.prompt.prompt_generator import prompt_for_label
from theseal.prompt.router import RoutingResult

from .schemas import AnswerResponse


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


# ---------------------------------------------------------------- LCEL 체인

def _classify_step(clf: Classifier | None):
    async def step(x: dict) -> RoutingResult | None:
        if x.get("label"):  # 라벨을 직접 주면 분류를 건너뛴다 (테스트 · 비교용)
            return None
        if clf is None:
            raise ClassifierNotReady("분류 모델을 불러오지 못했습니다")
        return await classify(clf, x["text"])
    return step


def prepare_chain(clf: Classifier | None) -> Runnable:
    """{"text", "label"?} → + classification · label · prompt (답변 직전까지)"""
    return (
        RunnablePassthrough.assign(classification=RunnableLambda(_classify_step(clf)))
        | RunnablePassthrough.assign(label=lambda x: x.get("label") or x["classification"].label)
        | RunnablePassthrough.assign(prompt=lambda x: prompt_for_label(x["label"], x["text"]))
    )


async def _answer_step(x: dict) -> AnswerResponse:
    result = await generate(x["prompt"], x.get("provider"), config=x.get("llm_config") or get_config())
    return AnswerResponse(**result.model_dump(), classification=x["classification"])


def answer_chain(clf: Classifier | None) -> Runnable:
    """{"text", "label"?, "provider"?} → AnswerResponse (AnswerResult + classification)"""
    return prepare_chain(clf) | RunnableLambda(_answer_step)


async def answer_stream(prompt: PromptSpec, provider: str | None, llm_config: LLMConfig) -> AsyncIterator[dict]:
    """delta 이벤트 … → done 이벤트. unknown(툴 경로)은 스트리밍하지 않고 done만 보낸다"""
    if prompt.label == "unknown":
        yield {"type": "done", "result": await generate(prompt, provider, config=llm_config)}
        return
    async for event in generate_stream(prompt, provider, config=llm_config):
        yield event
