# Seal_LLM

## 빠르게 실행해 보기

처음 받은 저장소에서 서버를 띄우고 답변까지 받아 보는 순서입니다.
분류 모델은 CPU로 1초 안에 학습되는 기준선(TF-IDF)을 쓰므로 GPU가 없어도 됩니다.

### 준비물

| 항목 | 용도 | 설치 |
| --- | --- | --- |
| Python 3.12 | 실행 환경 | `uv`가 맞는 버전을 알아서 준비합니다 |
| [uv](https://docs.astral.sh/uv/) | 패키지 · 가상환경 관리 | 공식 설치 안내 참고 |
| [Ollama](https://ollama.com/download) | 로컬 LLM (답변 생성 · LLM 검증) | 설치 후 실행해 두기 |
| 디스크 약 7GB | 로컬 LLM 약 6GB + 패키지 | |

### 1. 패키지 설치

```bash
uv sync
```

### 2. 환경 파일 만들기

```bash
cp .env.example .env
```

PowerShell은 `Copy-Item .env.example .env`입니다.
`.env`에는 비밀 키와 PC별 값만 둡니다. 키가 없어도 기본 기능은 동작합니다.

| 항목 | 없으면 |
| --- | --- |
| `OPENAI_API_KEY` | `provider: "openai"` 요청만 오류 (기본은 로컬 LLM) |
| `KMA_AUTH_KEY` | 날씨 질문에 기상청 대신 검색 툴이 답함 |
| `TAVILY_API_KEY` | 검색 툴이 빠짐 (날씨 · 상식 질문은 안내 문구로 답함) |

그 밖의 설정은 코드 기본값을 씁니다 (`src/theseal/common/config.py` 등).

### 3. 분류 모델 준비

가중치(`model/`)는 git에 없으므로 직접 학습합니다. 서버는
`src/theseal/classifier/config.py`의 `FINAL_MODEL`(현재 `baseline/20261013-1057`)이
가리키는 폴더를 씁니다.

```bash
uv run python scripts/classifier/train_baseline.py
```

마지막에 출력되는 `모델: runs/classifier/inquiries/baseline/<새 실행 ID>/model`을
`FINAL_MODEL`의 실행 ID 폴더로 옮깁니다. 기준선은 같은 설정이면 결과가 똑같아서
팀 최종 모델과 같은 모델입니다 (`val_report.json`의 점수로 확인할 수 있습니다).

```bash
mv runs/classifier/inquiries/baseline/<새 실행 ID>/model runs/classifier/inquiries/baseline/20261013-1057/model
```

PowerShell은 `Move-Item`을 씁니다.

### 4. 로컬 LLM 받기 · 연결 확인

Ollama가 실행 중이어야 합니다.

```bash
uv run scripts/answer/fetch_models.py
```

```bash
uv run scripts/answer/check_llm.py
```

`fetch_models.py`는 Gemma 4 E4B(약 6GB)를 받고 팀 기준 모델 ID인지 확인합니다.
`check_llm.py`는 Ollama 호출과 키 설정을 확인합니다. 마지막 줄이 `모두 정상`이면 준비 완료이고,
비어 있는 키는 `[-]`(선택 사항)로 표시됩니다. OpenAI · 기상청 · Tavily를 실제로
호출해 보려면 `--all`을 붙이세요 (OpenAI는 과금).

### 5. 서버 실행

```bash
uv run uvicorn --app-dir src theseal.api.main:app --reload --host 127.0.0.1 --port 8000
```

Windows에서는 `localhost` 대신 `127.0.0.1`로 접속하세요 (`localhost`는 요청마다 약 2초 늦습니다).
`.env`를 고친 뒤에는 서버를 다시 켜야 합니다.

### 6. 테스트

**상태 확인:** http://127.0.0.1:8000/health 에서 `classifier`가 `baseline/20261013-1057`이고
`classifier_error`가 `null`이면 준비 완료입니다.

**문서 화면에서 직접 호출 (추천):** http://127.0.0.1:8000/docs 에서 `POST /api/answer` →
**Try it out** → 아래 본문을 넣고 **Execute**.

```json
{"text": "주문한 상품이 일주일째 배송 준비중이에요. 언제 받을 수 있나요?"}
```

**터미널에서 호출 (macOS · Linux):**

```bash
curl -X POST http://127.0.0.1:8000/api/answer -H "Content-Type: application/json" -d '{"text": "비밀번호를 잊어버려서 로그인이 안 돼요."}'
```

**Windows:** 위 `curl` 명령은 쓰지 마세요. Git Bash에서는 한글이 UTF-8로 전달되지 않아
`400 There was an error parsing the body`가 나고, PowerShell에서는 `curl`이 PowerShell 명령
(`Invoke-WebRequest`)의 별칭이라 옵션 오류가 납니다. 문서 화면이나 아래 PowerShell 명령을 쓰세요.

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/answer -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"text": "비밀번호를 잊어버려서 로그인이 안 돼요."}'))
```

## 개발 환경 세팅

```bash
uv run scripts/classifier/fetch_models.py
```

`fetch_models.py`는 `models/` 아래에 분류기 후보 모델을 내려받습니다. 후보 목록은
`src/theseal/classifier/config.py`의 `MODEL_CANDIDATES`에 있습니다. 이미 받은 모델은
건너뛰므로 여러 번 돌려도 안전합니다. 다시 받으려면 해당 폴더를 지우고 실행하세요.


## 분류기

분류기를 불러와 쓰는 방법과 직접 학습하는 순서는
[src/theseal/classifier/README.md](src/theseal/classifier/README.md)에 있습니다.

### LLM (답변 생성)

로컬 LLM은 Ollama로 돌립니다. 준비와 연결 확인은 위 "빠르게 실행해 보기" 4단계를 따릅니다.

`scripts/answer/fetch_models.py`는 `OLLAMA_MODEL`을 Ollama로 내려받고, 모델 ID가 팀 기준과 같은지
확인합니다. 모델은 프로젝트 `models/`가 아니라 Ollama 저장소(`~/.ollama`)에 저장됩니다.

| 구분 | 모델 | 용량 | 준비 |
| --- | --- | --- | --- |
| 로컬 | hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M (ID `487eacf5e94b`) | 약 6GB | `fetch_models.py` |
| 상용 | OpenAI gpt-5.6-luna | - | `.env`에 `OPENAI_API_KEY` |
| 날씨 툴 | 기상청 API허브 단기예보 (지금 날씨 · 모레까지 예보) | - | `.env`에 `KMA_AUTH_KEY` (없으면 날씨 툴이 빠지고 검색 툴이 답함) |
| 검색 툴 | Tavily | - | `.env`에 `TAVILY_API_KEY` (없으면 검색 툴이 자동으로 빠짐) |

ID가 다르다는 경고가 나오면 Hugging Face 저장소 파일이 갱신된 것이므로 팀에 알려 주세요.

### 분류 후처리 (routing)

`src/theseal/prompt/`의 `router.py`는 분류기 결과를 받아 최종 라벨을 정합니다. 경로 규칙이 보정한 확률로
확신 · 확신 부족 · 접전 · 모름(unknown)을 판정하고, 확신이 아닌 문의는 LLM 검증이 라벨이 맞는지 한 번 더 확인합니다.
입력은 `Classifier.predict()` 결과(`ClassificationResult`) 하나입니다. 원문은 결과의 `text`, 라벨 이름은 task의 `labels.json` 순서,
보정값은 `FINAL_MODEL`의 것을 씁니다.

```python
clf = Classifier.from_final()
result = await router.run(clf.predict(text))  # 원문은 result.text
result.label  # 최종 라벨 (shipping · refund · account · unknown) → 프롬프트 생성
```

보정값(T · tau · tau_low · delta)은 학습한 가중치마다 다르므로 `src/theseal/prompt/routing_config.py`의
`ROUTE_CALIBRATIONS`에 모델별로 적어 둡니다. 키는 모델 폴더 이름(`<모델>/<실행 ID>`, 예전 구조는 `<모델>`)이고,
서비스는 쓰는 분류 모델의 키를 자동으로 찾습니다. 키가 없으면 예시 값을 씁니다 (`/api/answer` 응답 `classification`의
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
`/health`에 이유가 나옵니다. 모델 준비와 서버 실행은 위 "빠르게 실행해 보기" 3 · 5단계를 따릅니다.

| 메서드 | 경로 | 내용 |
| --- | --- | --- |
| GET | `/health` | 서버 상태 · 분류 모델 · 설정 |
| POST | `/api/answer` | `{"text", "provider"?, "label"?}` → 답변 + 분류 결과 (JSON) |
| POST | `/api/answer/stream` | 같은 요청 → SSE (`status` · `classification` · `delta` · `done`). 범위 밖(unknown)은 툴을 쓴 뒤 `done`만 보냄 |

`provider`는 `local` · `openai`(비우면 `DEFAULT_PROVIDER`, 기본 `local`), `label`을 넣으면 분류를 건너뜁니다.

