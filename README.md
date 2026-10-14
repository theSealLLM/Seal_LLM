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

### LLM (답변 생성)

로컬 LLM은 Ollama로 돌립니다. 먼저 [Ollama](https://ollama.com/download)를 설치하고 실행해 두세요.
모델 이름과 API 키는 `.env`에서 읽으므로 `.env.example`을 복사해 `.env`를 만들고 키를 채웁니다.

```bash
cp .env.example .env
```

```bash
uv run scripts/answer/fetch_models.py
```

`fetch_models.py`는 `.env`의 `OLLAMA_MODEL`을 Ollama로 내려받고, 모델 ID가 팀 기준과 같은지
확인합니다. 모델은 프로젝트 `models/`가 아니라 Ollama 저장소(`~/.ollama`)에 저장됩니다.

| 구분 | 모델 | 용량 | 준비 |
| --- | --- | --- | --- |
| 로컬 | hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M (ID `487eacf5e94b`) | 약 6GB | `fetch_models.py` |
| 상용 | OpenAI gpt-5.6-luna | - | `.env`에 `OPENAI_API_KEY` |
| 날씨 툴 | 기상청 API허브 단기예보 (지금 날씨 · 모레까지 예보) | - | `.env`에 `KMA_AUTH_KEY` (없으면 날씨 툴이 빠지고 검색 툴이 답함) |
| 검색 툴 | Tavily | - | `.env`에 `TAVILY_API_KEY` (없으면 검색 툴이 자동으로 빠짐) |

ID가 다르다는 경고가 나오면 Hugging Face 저장소 파일이 갱신된 것이므로 팀에 알려 주세요.

```bash
uv run scripts/answer/check_llm.py
```

`check_llm.py`는 Ollama 연결 · 모델 · 호출과 OpenAI · 기상청 · Tavily 키 설정을 한 번에 확인합니다.
OpenAI 호출(과금), 기상청 조회, Tavily 검색까지 확인하려면 `--all`을 붙이세요.

### 분류 후처리 (routing)

`src/theseal/routing/`은 분류기 결과를 받아 최종 라벨을 정합니다. 경로 규칙이 보정한 확률로
확신 · 확신 부족 · 접전 · 모름(unknown)을 판정하고, 확신이 아닌 문의는 LLM 검증이 라벨이 맞는지 한 번 더 확인합니다.
입력은 `Classifier.predict()` 결과 `(라벨 번호, 확률 배열)` 그대로이고, `Classifier.labels`로 이름을 붙입니다.

```python
clf = Classifier.from_final()
result = await router.run(clf.predict(text), text, labels=clf.labels, model_dir=clf.model_dir)
result.label  # 최종 라벨 (shipping · refund · account · unknown) → 프롬프트 생성
```

보정값(T · tau · tau_low · delta)은 학습한 가중치마다 다르므로 `src/theseal/prompt/routing_config.py`의
`ROUTE_CALIBRATIONS`에 모델별로 적어 둡니다. 키는 모델 폴더 이름(`<모델>/<실행 ID>`, 예전 구조는 `<모델>`)이고,
서비스는 쓰는 분류 모델의 키를 자동으로 찾습니다. 키가 없으면 예시 값을 씁니다 (`/api/classify` 응답의
`route.calibration_key`가 찾아본 키, `route.source`가 값의 출처).

켜고 끄기는 `.env`에서 합니다.

| 항목 | `.env` | 값 |
| --- | --- | --- |
| 경로 규칙 | `ROUTE_RULE_ENABLED` | `true` (기본) · `false` |
| LLM 검증 | `VERIFY_MODE` | `uncertain` (기본, 확신이 아닌 것만) · `all` (전부) · `off` |

둘 다 끄면 분류기 라벨을 그대로 씁니다. 새로 학습한 모델의 보정값은
`uv run scripts/classifier/calibrate_route_rule.py --model <모델> --run <실행 ID>`(최종 모델은 `--final`)로 만들고,
마지막에 출력되는 줄을 `ROUTE_CALIBRATIONS`에 붙여 넣습니다. 모델 폴더 옆 `route_rule.json`은 평가 수치를 담은
기록이고 서비스는 읽지 않습니다. `.env`의 `ROUTE_TAU` 등 개별 값을 적으면 모든 모델에 그 값이 먼저 적용됩니다.

### API 서버 (FastAPI)

분류 → 분류 후처리 → 프롬프트 → 답변 생성을 HTTP로 묶은 서버입니다. 분류 모델은 `classifier/config.py`의
`FINAL_MODEL`이 가리키는 `runs/classifier/<task>/<모델>/<실행 ID>/model/`을 씁니다. 모델이 없어도 서버는 뜨고,
`/health`에 이유가 나옵니다.

가중치(`model/`)는 git에 없으므로 처음에는 직접 만들어야 합니다. `FINAL_MODEL`과 같은 설정으로 학습한 뒤,
생긴 `model/` 폴더를 `FINAL_MODEL`의 실행 ID 폴더로 옮깁니다. 기준선(TF-IDF)은 같은 설정이면 결과가 똑같습니다.

```bash
uv run python scripts/classifier/train_baseline.py
```

예) 학습 결과가 `runs/classifier/inquiries/baseline/20261014-0900/model`이면
`runs/classifier/inquiries/baseline/20261013-1057/model`로 옮깁니다 (`val_report.json`의 점수가 같은지 확인).

```bash
uv run uvicorn --app-dir src theseal.api.main:app --reload --port 8000
```

| 메서드 | 경로 | 내용 |
| --- | --- | --- |
| GET | `/health` | 서버 상태 · 분류 모델 · 설정 |
| POST | `/api/classify` | `{"text"}` → 최종 라벨과 판정 근거 (분류기 확률 · 경로 · LLM 검증) |
| POST | `/api/answer` | `{"text", "provider"?, "label"?}` → 답변 + 분류 결과 (JSON) |
| POST | `/api/answer/stream` | 같은 요청 → SSE (`status` · `classification` · `delta` · `done`). 범위 밖(unknown)은 툴을 쓴 뒤 `done`만 보냄 |

`provider`는 `local` · `openai`(비우면 `.env`의 `DEFAULT_PROVIDER`), `label`을 넣으면 분류를 건너뜁니다.
문서 화면은 http://127.0.0.1:8000/docs 입니다. Windows에서는 `localhost`가 IPv6를 먼저 시도해 요청마다
약 2초 늦으므로 `127.0.0.1`을 쓰세요.

### OneDrive 주의

이 저장소가 OneDrive 동기화 폴더 안에 있으면 `uv sync`가 캐시에서 하드링크를
걸다 `os error 396`으로 실패합니다. `pyproject.toml`의 `[tool.uv]`에
`link-mode = "copy"`를 넣어 해결해 두었습니다.

`models/`는 gitignore 대상이지만 OneDrive는 그대로 업로드합니다. 1.75GB를
클라우드로 올리고 싶지 않다면 OneDrive 설정에서 `models/` 폴더를 동기화
제외하세요.
