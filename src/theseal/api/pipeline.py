"""API가 쓰는 기능 연결: 질문(text) → 분류기 → Prompt(분류 후처리 + 프롬프트) → 답변. 처리는 모두 Pipeline이 한다."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from theseal.answer.answer_generator import generate, generate_stream
from theseal.answer.config import AnswerConfig
from theseal.api import errors
from theseal.api.config import ApiConfig
from theseal.api.schemas import AnswerResponse
from theseal.classifier.classifier import ClassificationResult, Classifier
from theseal.classifier.config import DEFAULT_TASK, PROJECT_ROOT, FinalModelNotSetError, task_paths
from theseal.common.config import LLMConfig
from theseal.common.schemas import PromptSpec
from theseal.prompt import Prompt, RoutedPrompt


class ClassifierNotReadyError(RuntimeError):
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

    Args:
      classifier: 분류 모델. None이면 label을 직접 준 요청만 처리한다.
      prompt: 분류 후처리 + 프롬프트. 없으면 Prompt.from_env(llm_config) (Router 포함).
      classifier_error: 분류 모델을 불러오지 못한 이유 (/health · 오류 응답에 쓴다).
      llm_config: LLM 연결 · 공용 설정 (프롬프트 · 검증 · 답변이 같이 쓴다). 없으면 .env + 코드 기본값.
      answer_config: 답변 생성 설정 (툴 · 시간 제한). 없으면 .env + 코드 기본값.
    """

    def __init__(self, classifier: Classifier | None = None, prompt: Prompt | None = None, *,
                 classifier_error: str | None = None, llm_config: LLMConfig | None = None,
                 answer_config: AnswerConfig | None = None) -> None:
        self.classifier = classifier
        self.classifier_error = classifier_error
        self.llm_config = llm_config or LLMConfig.from_env()
        self.answer_config = answer_config or AnswerConfig.from_env()
        self.prompt = prompt or Prompt.from_env(llm_config=self.llm_config)

    @classmethod
    def from_env(cls, config: ApiConfig | None = None) -> Pipeline:
        """서비스가 쓰는 Pipeline: FINAL_MODEL 분류 모델을 불러온다. 없으면 이유를 남기고 분류기 없이 만든다."""
        try:
            classifier = load_classifier(config or ApiConfig.from_env())
        except Exception as e:  # noqa: BLE001 — 모델이 없어도 서버는 띄운다. 이유는 lifespan이 로그로 남긴다
            return cls(classifier_error=f"{type(e).__name__}: {e}")
        return cls(classifier)

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

    async def answer(self, text: str, label: str | None, provider: str | None) -> AnswerResponse:
        """질문 하나 → AnswerResponse (/api/answer). 분류 모델 없이 분류가 필요하면 config_error 응답."""
        if label is None and self.classifier is None:
            return errors.classifier_missing(self.llm_config, self.classifier_error, provider)
        chain_input = {"text": text, "label": label, "provider": provider}
        return await self.answer_chain().ainvoke(chain_input)

    def status(self) -> dict[str, Any]:
        """서버 상태 · 분류 모델 · 분류 후처리 설정 · 기본 provider (/health)."""
        clf, router = self.classifier, self.prompt.router
        return {
            "status": "ok",
            "classifier": classifier_name(clf) if clf else None,
            "classifier_device": str(getattr(clf.model, "device", "cpu")) if clf else None,
            "classifier_error": self.classifier_error,
            "route_rule_enabled": router.config.route_rule_enabled if router else None,
            "verify_mode": router.config.verify_mode if router else None,
            "default_provider": self.llm_config.default_provider,
        }

    async def answer_stream(self, prompt: PromptSpec,
                            provider: str | None) -> AsyncIterator[dict[str, Any]]:
        """delta 이벤트 … → done 이벤트. unknown(툴 경로)은 스트리밍하지 않고 done만 보낸다"""
        configs = {"config": self.llm_config, "answer_config": self.answer_config}
        if prompt.label == "unknown":
            yield {"type": "done", "result": await generate(prompt, provider, **configs)}
            return
        async for event in generate_stream(prompt, provider, **configs):
            yield event

    async def _predict(self, x: dict[str, Any]) -> ClassificationResult | None:
        if x.get("label"):  # 라벨을 직접 주면 분류를 건너뛴다 (테스트 · 비교용)
            return None
        if self.classifier is None:
            raise ClassifierNotReadyError("분류 모델을 불러오지 못했습니다")
        return await run_in_threadpool(self.classifier.predict, x["text"])

    async def _route(self, x: dict[str, Any]) -> RoutedPrompt:
        if x["prediction"] is None:
            return self.prompt.build_for_label(x["label"], x["text"])
        return await self.prompt.build_with_routing(x["prediction"])

    async def _answer(self, x: dict[str, Any]) -> AnswerResponse:
        result = await generate(x["prompt"], x.get("provider"), config=self.llm_config,
                                answer_config=self.answer_config)
        return AnswerResponse(**result.model_dump(), classification=x["classification"])


def classifier_dir(name: str = "final", task: str = DEFAULT_TASK) -> Path:
    """분류 모델 폴더. API는 항상 final(FINAL_MODEL)을 쓴다.
    로컬 평가 스크립트는 "<모델>/<실행 ID>"나 모델 폴더 경로로 다른 모델도 고를 수 있다."""
    paths = task_paths(task)
    if name == "final":
        try:
            return paths.final_model_dir
        except FinalModelNotSetError as e:
            raise ClassifierNotReadyError(str(e)) from None
    model, sep, run = name.partition("/")
    if sep and model and run and "/" not in run and not Path(name).exists():
        return paths.model_dir(model, run)
    path = Path(name)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_classifier(config: ApiConfig, name: str = "final") -> Classifier:
    """분류 모델을 불러온다. 폴더가 없거나 FINAL_MODEL이 비어 있으면 ClassifierNotReadyError."""
    model_dir = classifier_dir(name, config.api_classifier_task)
    if not model_dir.exists():
        raise ClassifierNotReadyError(
            f"분류 모델이 없습니다: {model_dir} (FINAL_MODEL과 같은 설정으로 train_*.py를 돌려 이 폴더에 두거나 "
            "share_model.py pull로 받으세요)")
    return Classifier(model_dir, device=config.api_classifier_device)


def classifier_name(clf: Classifier) -> str:
    """<모델>/<실행 ID> (모델 폴더의 위 두 단계)"""
    return f"{clf.model_dir.parent.parent.name}/{clf.model_dir.parent.name}"


# E2E · 평가 스크립트가 부르는 함수 (Pipeline을 그때그때 만든다)
def prepare_chain(clf: Classifier | None) -> Runnable:
    """`Pipeline(clf).prepare_chain()`의 단축 함수."""
    return Pipeline(clf).prepare_chain()


def answer_chain(clf: Classifier | None) -> Runnable:
    """`Pipeline(clf).answer_chain()`의 단축 함수."""
    return Pipeline(clf).answer_chain()
