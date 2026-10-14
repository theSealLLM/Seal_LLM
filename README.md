# Seal_LLM

## 개발 환경 세팅

```bash
uv sync
```

```bash
uv run scripts/classifier/fetch_models.py
```

`fetch_models.py`는 `models/` 아래에 분류기 후보 모델을 내려받습니다. 후보 목록은
`src/theseal/classifier/config.py`의 `MODEL_CANDIDATES`에 있습니다. 이미 받은 모델은
건너뛰므로 여러 번 돌려도 안전합니다. 다시 받으려면 해당 폴더를 지우고 실행하세요.


## 분류기

분류기를 불러와 쓰는 방법과 직접 학습하는 순서는
[src/theseal/classifier/README.md](src/theseal/classifier/README.md)에 있습니다.
