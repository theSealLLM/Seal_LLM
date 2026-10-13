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
uv run scripts/llm/fetch_models.py
```

`fetch_models.py`는 `.env`의 `OLLAMA_MODEL`을 Ollama로 내려받고, 모델 ID가 팀 기준과 같은지
확인합니다. 모델은 프로젝트 `models/`가 아니라 Ollama 저장소(`~/.ollama`)에 저장됩니다.

| 구분 | 모델 | 용량 | 준비 |
| --- | --- | --- | --- |
| 로컬 | hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M (ID `487eacf5e94b`) | 약 6GB | `fetch_models.py` |
| 상용 | OpenAI gpt-5.6-luna | - | `.env`에 `OPENAI_API_KEY` |
| 검색 툴 | Tavily | - | `.env`에 `TAVILY_API_KEY` (없으면 검색 툴이 자동으로 빠짐) |

ID가 다르다는 경고가 나오면 Hugging Face 저장소 파일이 갱신된 것이므로 팀에 알려 주세요.

```bash
uv run scripts/llm/check_llm.py
```

`check_llm.py`는 Ollama 연결 · 모델 · 호출과 OpenAI · Tavily 키 설정을 한 번에 확인합니다.
OpenAI 호출(과금)과 Tavily 검색까지 확인하려면 `--all`을 붙이세요.

### OneDrive 주의

이 저장소가 OneDrive 동기화 폴더 안에 있으면 `uv sync`가 캐시에서 하드링크를
걸다 `os error 396`으로 실패합니다. `pyproject.toml`의 `[tool.uv]`에
`link-mode = "copy"`를 넣어 해결해 두었습니다.

`models/`는 gitignore 대상이지만 OneDrive는 그대로 업로드합니다. 1.75GB를
클라우드로 올리고 싶지 않다면 OneDrive 설정에서 `models/` 폴더를 동기화
제외하세요.
