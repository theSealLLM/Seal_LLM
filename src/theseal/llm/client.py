"""LLM 연결: provider(local · openai)에 맞는 LangChain 채팅 모델을 만든다.

LLM 분기는 이 파일 한 곳에만 둔다. 답변 생성(answer/)과 LLM 검증(routing/)이 함께 쓴다.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import PROVIDERS, LLMConfig


class ConfigError(Exception):
    """키 · 모델 이름 등 설정이 없어 LLM을 만들 수 없음 → AnswerResult의 config_error"""


def model_name(provider: str, config: LLMConfig) -> str:
    return config.ollama_model if provider == "local" else config.openai_model


def llm_options(provider: str, config: LLMConfig, *, with_tools: bool = False) -> dict:
    """AnswerResult.llm_options에 남길 생성 조건."""
    if provider == "local":
        reasoning = config.ollama_reasoning_with_tools if with_tools else config.ollama_reasoning
        return {"reasoning": reasoning, "with_tools": with_tools}
    return {"reasoning_effort": config.openai_reasoning_effort, "with_tools": with_tools}


def make_llm(provider: str, config: LLMConfig, *, with_tools: bool = False) -> BaseChatModel:
    """with_tools: 툴 경로에서 쓸 LLM인지. 로컬 모델의 thinking 설정을 경로별로 고른다."""
    if provider not in PROVIDERS:
        raise ConfigError(f"알 수 없는 provider: {provider!r} (가능: {', '.join(PROVIDERS)})")

    if provider == "local":
        from langchain_ollama import ChatOllama

        if not config.ollama_model:
            raise ConfigError("OLLAMA_MODEL이 설정되지 않았습니다.")
        reasoning = config.ollama_reasoning_with_tools if with_tools else config.ollama_reasoning
        keep_alive: str | int | None = config.ollama_keep_alive or None
        if keep_alive and keep_alive.lstrip("-").isdigit():  # "-1", "3600" → 초, "30m" → 그대로
            keep_alive = int(keep_alive)
        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            reasoning=reasoning,
            keep_alive=keep_alive,
        )

    from langchain_openai import ChatOpenAI

    if not config.openai_api_key:
        raise ConfigError("OPENAI_API_KEY가 설정되지 않았습니다.")
    if not config.openai_model:
        raise ConfigError("OPENAI_MODEL이 설정되지 않았습니다.")
    kwargs = {}
    if config.openai_reasoning_effort:
        kwargs["reasoning_effort"] = config.openai_reasoning_effort
    # gpt-5.6-luna는 /v1/chat/completions에서 추론과 툴을 함께 쓰지 못한다 → Responses API
    return ChatOpenAI(
        model=config.openai_model,
        api_key=config.openai_api_key,
        timeout=config.llm_timeout_sec,
        max_retries=1,
        use_responses_api=True,
        **kwargs,
    )
