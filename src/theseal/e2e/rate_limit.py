"""OpenAI 호출 속도 조절: 분당 요청 수 상한 + 한도 초과(429) · 일시 오류면 기다렸다가 다시 보낸다.

1000건을 한꺼번에 보내면 계정 등급(tier)의 분당 요청 · 토큰 한도(RPM · TPM)에 걸려 429(RateLimitError)가 난다.
    - 호출 시작 간격을 60/rpm초 이상으로 벌린다. 동시에 기다리는 호출 수는 runner의 concurrency가 따로 막는다
    - 429 · 시간 초과 · 연결 오류 · 5xx면 점점 길게(5 → 10 → 20초) 기다렸다가 최대 3번 다시 보낸다.
      그동안 다른 호출도 같이 쉰다. 3번 다 실패하면 그 문의는 실패로 넘기고, 문의가 연달아 실패하면 runner가 멈춘다
    - 크레딧 소진(insufficient_quota) · 키 오류는 기다려도 풀리지 않으므로 바로 멈춘다 (OpenAIStopped)
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

# 오류 문자열에 이 말이 있으면 다시 보낸다. answer_generator.generate는 예외 대신
# AnswerResult.error.message에 "RateLimitError: Error code: 429 ..."처럼 예외 이름을 남긴다
RETRYABLE = ("RateLimitError", "rate_limit", "APITimeoutError", "APIConnectionError", "InternalServerError",
             "TimeoutError", "시간 초과", "Error code: 5")
# 기다려도 안 풀리는 오류
FATAL = ("insufficient_quota", "AuthenticationError", "invalid_api_key", "PermissionDeniedError")


class OpenAIStopped(Exception):
    """크레딧 소진 · 키 오류. 이 단계를 멈추고 설정을 고친 뒤 다시 실행한다."""


class RateLimiter:
    """기본값은 직접 만들 때만 쓴다. 실행 설정은 scripts/e2e/run_e2e.py 상단(OPENAI_*)에서 넘긴다."""

    def __init__(self, rpm: float, *, max_retries: int = 3, base_delay: float = 5, max_delay: float = 120):
        self.interval = 60 / rpm if rpm > 0 else 0
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._next = 0.0  # 다음 호출을 시작해도 되는 시각 (time.monotonic)
        self._lock = asyncio.Lock()

    async def _wait_turn(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            await asyncio.sleep(start - now)

    async def call(self, fn: Callable[[], Awaitable[T]], *, error_of: Callable[[T], str | None] = lambda r: None,
                   label: str = "") -> T:
        """fn()을 차례가 오면 부른다. 예외가 나거나 error_of(결과)가 문자열이면 오류로 보고 다시 보낼지 정한다."""
        for attempt in range(self.max_retries + 1):
            await self._wait_turn()
            try:
                result = await fn()
                error = error_of(result)
            except Exception as e:  # noqa: BLE001 — 오류 종류를 보고 다시 보낼지 정한다
                result, error = None, f"{type(e).__name__}: {e}"
            if error is None:
                return result
            if any(key in error for key in FATAL):
                raise OpenAIStopped(error)
            if attempt == self.max_retries or not any(key in error for key in RETRYABLE):
                raise RuntimeError(error)
            delay = min(self.base_delay * 2 ** attempt, self.max_delay) * random.uniform(0.75, 1.25)
            self._next = max(self._next, time.monotonic() + delay)  # 다른 호출도 같이 쉬게 한다
            print(f"  [OpenAI 대기] {label} {error[:120]} → {delay:.0f}초 뒤 다시 "
                  f"({attempt + 1}/{self.max_retries})", flush=True)
        raise AssertionError("unreachable")
