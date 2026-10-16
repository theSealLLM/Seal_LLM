"""API가 쓰는 기능 연결: 질문(text) → 분류기 → Prompt(분류 후처리 + 프롬프트) → 답변. 처리는 모두 Pipeline이 한다."""

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
from theseal.prompt import Prompt, RoutedPrompt

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


class Pipeline:
    """분류기 → Prompt → 답변을 LCEL 체인으로 잇는다. 단계마다 RunnablePassthrough.assign으로 결과를 쌓아서,
    최종 응답에 분류 근거(classification)가 남는다. 분류 후처리(LLM 검증 포함)는 Prompt 안에서 한 번만 돈다.

        {"text", "label"?, "provider"?}
          ① prediction      분류기 classifier.predict → ClassificationResult (label을 주면 건너뜀)
          ② routed          Prompt.build_with_routing → RoutedPrompt(routing, prompt)
                            (label을 주면 Prompt.build_for_label)
             classification · label · prompt   routed를 풀어 둔다 (스트리밍 · 응답에서 쓰기 쉽게)
          ③ AnswerResponse  generate → AnswerResult + classification

    예) pipeline = Pipeline.from_env()          # 서버 시작 때 한 번 (분류 모델이 없어도 만든다)
        response = await pipeline.answer_chain().ainvoke({"text": "환불은 언제 되나요?"})
    스트리밍은 prompt까지 prepare_chain으로 만들고 답변만 answer_stream으로 흘린다.
    """

    def __init__(self, classifier: Classifier | None = None, prompt: Prompt | None = None, *,
                 classifier_error: str | None = None, llm_config: LLMConfig | None = None) -> None:
        """classifier: 분류 모델. None이면 label을 직접 준 요청만 처리한다
        classifier_error: 분류 모델을 불러오지 못한 이유 (/health · 오류 응답에 쓴다)
        prompt: 분류 후처리 + 프롬프트. 없으면 Prompt() (Router는 처음 쓸 때 만든다)
        llm_config: 답변 생성 설정. 없으면 .env + 코드 기본값"""
        self.classifier = classifier
        self.classifier_error = classifier_error
        self.prompt = prompt or Prompt()
        self.llm_config = llm_config

    @classmethod
    def from_env(cls, config: ApiConfig | None = None) -> Pipeline:
        """서비스가 쓰는 Pipeline: FINAL_MODEL 분류 모델을 불러온다. 없으면 이유를 남기고 분류기 없이 만든다."""
        try:
            return cls(load_classifier(config or ApiConfig.from_env()))
        except Exception as e:  # noqa: BLE001 — 모델이 없어도 서버는 띄운다 (label을 직접 넣은 요청은 동작)
            return cls(classifier_error=f"{type(e).__name__}: {e}")

    def prepare_chain(self) -> Runnable:
        """{"text", "label"?} → + prediction · routed · classification · label · prompt (답변 직전까지)"""
        return (
            RunnablePassthrough.assign(prediction=RunnableLambda(self._predict))  # ① 분류기
            | RunnablePassthrough.assign(routed=RunnableLambda(self._route))  # ② Prompt
            | RunnablePassthrough.assign(
                classification=lambda x: x["routed"].routing,
                label=lambda x: x["routed"].label,
                prompt=lambda x: x["routed"].prompt,
            )
        )

    def answer_chain(self) -> Runnable:
        """{"text", "label"?, "provider"?} → AnswerResponse (AnswerResult + classification)"""
        return self.prepare_chain() | RunnableLambda(self._answer)

    async def answer_stream(self, prompt: PromptSpec, provider: str | None) -> AsyncIterator[dict]:
        """delta 이벤트 … → done 이벤트. unknown(툴 경로)은 스트리밍하지 않고 done만 보낸다"""
        config = self.llm_config or get_config()
        if prompt.label == "unknown":
            yield {"type": "done", "result": await generate(prompt, provider, config=config)}
            return
        async for event in generate_stream(prompt, provider, config=config):
            yield event

    async def _predict(self, x: dict):
        if x.get("label"):  # 라벨을 직접 주면 분류를 건너뛴다 (테스트 · 비교용)
            return None
        if self.classifier is None:
            raise ClassifierNotReady("분류 모델을 불러오지 못했습니다")
        return await run_in_threadpool(self.classifier.predict, x["text"])

    async def _route(self, x: dict) -> RoutedPrompt:
        if x["prediction"] is None:
            return self.prompt.build_for_label(x["label"], x["text"])
        return await self.prompt.build_with_routing(x["prediction"])

    async def _answer(self, x: dict) -> AnswerResponse:
        result = await generate(x["prompt"], x.get("provider"), config=self.llm_config or get_config())
        return AnswerResponse(**result.model_dump(), classification=x["classification"])


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


# E2E · 평가 스크립트가 부르는 함수 (Pipeline을 그때그때 만든다)
def prepare_chain(clf: Classifier | None) -> Runnable:
    return Pipeline(clf).prepare_chain()


def answer_chain(clf: Classifier | None) -> Runnable:
    return Pipeline(clf).answer_chain()
