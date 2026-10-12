# Seal_LLM

## 개발 환경 세팅

```bash
uv sync
```

```bash
uv run scripts/classifier/fetch_models.py
```

`fetch_models.py`는 `models/` 아래에 분류기 후보 모델 3종을 내려받습니다.

| 모델 | 용량 |
| --- | --- |
| klue/bert-base | 850MB |
| klue/roberta-base | 846MB |
| monologg/koelectra-small-v3-discriminator | 55MB |

합계 약 1.75GB입니다. 이미 받은 모델은 건너뛰므로 여러 번 돌려도 안전합니다.
다시 받으려면 해당 폴더를 지우고 실행하세요.

### OneDrive 주의

이 저장소가 OneDrive 동기화 폴더 안에 있으면 `uv sync`가 캐시에서 하드링크를
걸다 `os error 396`으로 실패합니다. `pyproject.toml`의 `[tool.uv]`에
`link-mode = "copy"`를 넣어 해결해 두었습니다.

`models/`는 gitignore 대상이지만 OneDrive는 그대로 업로드합니다. 1.75GB를
클라우드로 올리고 싶지 않다면 OneDrive 설정에서 `models/` 폴더를 동기화
제외하세요.
