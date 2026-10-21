"""기상청 날씨 툴 (기상청 API허브 · 단기예보 조회서비스). 지금 날씨와 오늘~모레 예보를 함께 돌려준다.

- API: https://apihub.kma.go.kr  /api/typ02/openApi/VilageFcstInfoService_2.0
    getUltraSrtNcst  초단기실황 (지금 날씨: 기온 · 습도 · 1시간 강수량 · 강수 형태 · 풍속)
    getVilageFcst    단기예보 (오늘~모레: 최저 · 최고 기온, 하늘 상태, 강수 형태, 강수 확률)
- 인증키 · 주소: AnswerConfig의 kma_auth_key(.env의 KMA_AUTH_KEY) · kma_api_base.
  키가 없으면 레지스트리가 이 툴을 등록하지 않는다
- 툴은 WeatherClient.from_config(config).tool()로 만든다. 키를 들고 있는 클라이언트가 툴을 실행한다
- 지역: 아래 REGIONS 표(17개 시 · 도 + 주요 도시)만 지원한다. 위경도를 기상청 격자(nx, ny)로 바꿔 조회한다.
  표에 없는 지역은 ValueError → tool_caller가 실패로 기록하고, 프롬프트 지시에 따라 검색 툴로 넘어간다
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx
from langchain_core.tools import BaseTool, StructuredTool

from theseal.answer.config import KMA_API_BASE, AnswerConfig
from theseal.answer.tools.tool_date import WEEKDAYS_KO
from theseal.common.config import TIMEZONE

# 지역 이름 → (위도, 경도). 시 · 도는 대표 도시(도청 또는 주요 시청) 위치
REGIONS: dict[str, tuple[float, float]] = {
    # 17개 시 · 도
    "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756), "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052), "광주": (35.1595, 126.8526), "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114), "세종": (36.4800, 127.2890), "경기": (37.2636, 127.0286),
    "강원": (37.8813, 127.7298), "충북": (36.6424, 127.4890), "충남": (36.6588, 126.6728),
    "전북": (35.8242, 127.1480), "전남": (34.8161, 126.4629), "경북": (36.5760, 128.5056),
    "경남": (35.2280, 128.6811), "제주": (33.4996, 126.5312),
    # 주요 도시
    "수원": (37.2636, 127.0286), "성남": (37.4200, 127.1265), "고양": (37.6584, 126.8320),
    "용인": (37.2411, 127.1776), "부천": (37.5034, 126.7660), "안산": (37.3219, 126.8309),
    "안양": (37.3943, 126.9568), "남양주": (37.6360, 127.2165), "화성": (37.1995, 126.8312),
    "평택": (36.9921, 127.1129), "의정부": (37.7381, 127.0337), "파주": (37.7599, 126.7800),
    "김포": (37.6152, 126.7156), "춘천": (37.8813, 127.7298), "원주": (37.3422, 127.9202),
    "강릉": (37.7519, 128.8761), "속초": (38.2070, 128.5918), "청주": (36.6424, 127.4890),
    "충주": (36.9910, 127.9259), "천안": (36.8151, 127.1139), "아산": (36.7898, 127.0019),
    "전주": (35.8242, 127.1480), "군산": (35.9676, 126.7369), "목포": (34.8118, 126.3922),
    "여수": (34.7604, 127.6622), "순천": (34.9507, 127.4872), "포항": (36.0190, 129.3435),
    "경주": (35.8562, 129.2247), "구미": (36.1195, 128.3446), "안동": (36.5684, 128.7294),
    "창원": (35.2280, 128.6811), "김해": (35.2285, 128.8894), "진주": (35.1800, 128.1076),
    "통영": (34.8544, 128.4331), "거제": (34.8806, 128.6211), "서귀포": (33.2541, 126.5600),
}
ALIASES = {
    "충청북도": "충북", "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북",
    "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "강원특별자치도": "강원",
    "제주특별자치도": "제주", "제주도": "제주", "세종특별자치시": "세종",
}
SUFFIXES = ("특별자치시", "특별자치도", "특별시", "광역시", "도", "시", "군")

SKY = {"1": "맑음", "3": "구름많음", "4": "흐림"}
PTY = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈", "4": "소나기",
       "5": "빗방울", "6": "빗방울눈날림", "7": "눈날림"}
FCST_ISSUE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)  # 단기예보 발표 시각
NCST_READY_MIN = 40  # 초단기실황은 정시 자료를 이 분 이후에 조회한다 (여유 포함)
FCST_READY_MIN = 15  # 단기예보는 발표 후 이 분 이후에 조회한다 (여유 포함)
HTTP_TIMEOUT_SEC = 8

WEATHER_TOOL_NAME = "get_weather"
# LLM에 보내는 툴 설명. 예전 @tool 함수의 docstring 그대로다 (줄 앞 공백 4칸까지 같아야 LLM 입력이 바뀌지 않는다)
WEATHER_TOOL_DESCRIPTION = (
    "기상청 자료로 지금 날씨와 오늘 · 내일 · 모레 예보를 알려 준다. 날씨 · 기온 · 비 · 눈 질문에 먼저 쓴다.\n"
    "    location은 시 · 도 또는 주요 도시 이름 (예: 서울, 부산, 제주, 수원). 지역을 말하지 않았으면 '서울'.\n"
    "    am은 06~12시, pm은 12~18시. pop_max_pct는 강수 확률 최댓값(%)."
)


# ---------------------------------------------------------------- 지역 · 격자

def resolve_region(location: str) -> str:
    """'서울특별시', '부산 해운대구', '경기도' → 표의 지역 이름. 없으면 ValueError."""
    text = (location or "").strip()
    candidates = [text.replace(" ", "")] + text.split()
    for cand in candidates:
        name = ALIASES.get(cand, cand)
        if name in REGIONS:
            return name
        for suffix in SUFFIXES:
            if name.endswith(suffix) and name[: -len(suffix)] in REGIONS:
                return name[: -len(suffix)]
    raise ValueError(f"지원하지 않는 지역입니다: {location!r} (시 · 도 또는 주요 도시만 지원)")


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """위경도 → 기상청 동네예보 격자(nx, ny). 기상청 공개 변환식(Lambert Conformal Conic)."""
    re_, grid = 6371.00877, 5.0
    slat1, slat2, olon, olat = (math.radians(v) for v in (30.0, 60.0, 126.0, 38.0))
    xo, yo = 43, 136
    re = re_ / grid
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
        math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1) / sn
    ro = re * sf / math.tan(math.pi * 0.25 + olat * 0.5) ** sn
    ra = re * sf / math.tan(math.pi * 0.25 + math.radians(lat) * 0.5) ** sn
    theta = math.radians(lon) - olon
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi
    theta *= sn
    return math.floor(ra * math.sin(theta) + xo + 0.5), math.floor(ro - ra * math.cos(theta) + yo + 0.5)


# ---------------------------------------------------------------- 발표 시각

def now_kst() -> datetime:
    """서비스 시간대(TIMEZONE) 기준 지금 시각을 돌려준다. WeatherClient의 기본 시계다."""
    return datetime.now(TIMEZONE)


def ncst_base(now: datetime) -> tuple[str, str]:
    """초단기실황: 매시 정시 자료. NCST_READY_MIN분 전이면 한 시간 앞 자료를 쓴다."""
    t = now if now.minute >= NCST_READY_MIN else now - timedelta(hours=1)
    return t.strftime("%Y%m%d"), t.strftime("%H00")


def fcst_base(now: datetime) -> tuple[str, str]:
    """단기예보: 가장 최근 발표(02 · 05 · … · 23시) 중 FCST_READY_MIN분이 지난 것."""
    t = now - timedelta(minutes=FCST_READY_MIN)
    hours = [h for h in FCST_ISSUE_HOURS if h <= t.hour]
    if hours:
        return t.strftime("%Y%m%d"), f"{hours[-1]:02d}00"
    prev = t - timedelta(days=1)
    return prev.strftime("%Y%m%d"), "2300"


# ---------------------------------------------------------------- 해석

def summarize_now(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """초단기실황 item 목록을 지금 날씨 요약(기온 · 습도 · 1시간 강수량 · 강수 형태 · 풍속)으로 돌려준다."""
    v = {it["category"]: it["obsrValue"] for it in items}
    first = items[0] if items else {}
    return {
        "observed_at": f"{first.get('baseDate', '')} {first.get('baseTime', '')}",
        "temperature_c": _num(v.get("T1H")),
        "humidity_pct": _num(v.get("REH")),
        "rain_1h_mm": _num(v.get("RN1")),
        "precipitation": PTY.get(str(v.get("PTY", "0")), "알 수 없음"),
        "wind_ms": _num(v.get("WSD")),
    }


def summarize_forecast(items: Sequence[Mapping[str, Any]], today: datetime) -> list[dict[str, Any]]:
    """날짜별(오늘 · 내일 · 모레) 최저 · 최고 기온과 오전 · 오후의 하늘 · 강수 · 강수 확률을 돌려준다.

    Args:
      items: 단기예보 item 목록.
      today: 기준 시각. 이 날짜부터 사흘을 본다.

    Returns:
      자료가 있는 날만 오늘 · 내일 · 모레 순서로 담는다.
    """
    days = []
    for offset, name in enumerate(("오늘", "내일", "모레")):
        d = today + timedelta(days=offset)
        key = d.strftime("%Y%m%d")
        rows = [it for it in items if it["fcstDate"] == key]
        if not rows:
            continue
        temps = [_num(it["fcstValue"]) for it in rows if it["category"] == "TMP"]
        tmn = next((_num(it["fcstValue"]) for it in rows if it["category"] == "TMN"), None)
        tmx = next((_num(it["fcstValue"]) for it in rows if it["category"] == "TMX"), None)
        day = {"day": name, "date": d.strftime("%Y-%m-%d"), "weekday": WEEKDAYS_KO[d.weekday()],
               "min_c": tmn if tmn is not None else (min(temps) if temps else None),
               "max_c": tmx if tmx is not None else (max(temps) if temps else None)}
        for part, hours in (("am", range(6, 12)), ("pm", range(12, 18))):
            part_rows = [it for it in rows if int(it["fcstTime"][:2]) in hours
                         and it["category"] in ("SKY", "PTY", "POP")]
            sky = Counter(it["fcstValue"] for it in part_rows if it["category"] == "SKY").most_common(1)
            ptys = [it["fcstValue"] for it in part_rows if it["category"] == "PTY" and it["fcstValue"] != "0"]
            pops = [_num(it["fcstValue"]) for it in part_rows if it["category"] == "POP"]
            day[part] = {
                "sky": SKY.get(sky[0][0], "알 수 없음") if sky else None,
                "precipitation": PTY.get(Counter(ptys).most_common(1)[0][0]) if ptys else "없음",
                "pop_max_pct": max(pops) if pops else None,
            } if part_rows else None
        days.append(day)
    return days


def _num(value: str | None) -> int | float | str | None:
    """기상청 값 문자열 → 숫자. 정수로 떨어지면 int."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value  # "강수없음" 같은 문자열은 그대로
    return int(f) if f.is_integer() else f


# ---------------------------------------------------------------- 호출

@dataclass(frozen=True)
class WeatherClient:
    """기상청 단기예보 API로 날씨를 조회하는 클라이언트. tool()이 LLM에 붙일 get_weather 툴을 만든다.

        tool = WeatherClient.from_config(answer_config).tool()
        tool.invoke({"location": "서울"})  →  {"location": "서울", "now": {…}, "forecast": [{…}, …]}
    """

    auth_key: str = field(repr=False)
    """기상청 API허브 인증키. 요청 쿼리(authKey)에 들어가므로 repr · 오류 메시지에 남기지 않는다"""

    api_base: str = KMA_API_BASE
    _: KW_ONLY

    http_get: Callable[..., httpx.Response] = httpx.get
    """HTTP GET 함수. 테스트에서 가짜 응답을 끼울 때 바꾼다"""

    now: Callable[[], datetime] = now_kst
    """지금 시각. 테스트에서 고정할 때 바꾼다"""

    @classmethod
    def from_config(cls, config: AnswerConfig) -> WeatherClient:
        """인증키 · 주소를 AnswerConfig(KMA_AUTH_KEY · KMA_API_BASE)에서 읽는다."""
        return cls(auth_key=config.kma_auth_key, api_base=config.kma_api_base)

    def tool(self) -> BaseTool:
        """LLM에 붙일 get_weather 툴. 실행하면 이 클라이언트의 get_weather를 부른다."""
        return StructuredTool.from_function(self.get_weather, name=WEATHER_TOOL_NAME,
                                            description=WEATHER_TOOL_DESCRIPTION)

    def get_weather(self, location: str) -> dict[str, Any]:
        """지역의 지금 날씨와 오늘 · 내일 · 모레 예보를 돌려준다. 지원하지 않는 지역은 ValueError."""
        region = resolve_region(location)
        nx, ny = latlon_to_grid(*REGIONS[region])
        now = self.now()
        result = {"location": region, "grid": {"nx": nx, "ny": ny}, "source": "기상청 단기예보"}
        result["now"] = summarize_now(self.fetch_items("getUltraSrtNcst", *ncst_base(now), nx, ny))
        base_date, base_time = fcst_base(now)
        result["forecast_issued"] = f"{base_date} {base_time}"
        items = self.fetch_items("getVilageFcst", base_date, base_time, nx, ny)
        result["forecast"] = summarize_forecast(items, now)
        return result

    def fetch_items(self, operation: str, base_date: str, base_time: str, nx: int,
                    ny: int) -> list[dict[str, Any]]:
        """기상청 API 하나를 불러 item 목록을 돌려준다. API허브 · HTTP · 결과 코드 오류는 RuntimeError."""
        params = {"pageNo": 1, "numOfRows": 1500, "dataType": "JSON", "base_date": base_date,
                  "base_time": base_time, "nx": nx, "ny": ny, "authKey": self.auth_key}
        resp = self.http_get(f"{self.api_base}/{operation}", params=params, timeout=HTTP_TIMEOUT_SEC)
        # 오류 메시지에 URL을 넣지 않는다 (쿼리에 authKey가 있고, 툴 오류는 응답 tools_used로 나간다)
        try:
            data = resp.json()
        except ValueError:  # 5xx HTML 페이지 등
            raise RuntimeError(f"기상청 API 응답이 JSON이 아닙니다 (HTTP {resp.status_code})") from None
        if "result" in data:  # API허브 자체 오류 (인증키 등). HTTP 오류여도 이 메시지가 더 구체적이다
            result = data["result"]
            raise RuntimeError(f"기상청 API 오류 {result.get('status')}: {result.get('message')}")
        if resp.is_error:
            raise RuntimeError(f"기상청 API HTTP 오류 {resp.status_code}")
        header = data["response"]["header"]
        if header.get("resultCode") != "00":
            raise RuntimeError(f"기상청 API 오류 {header.get('resultCode')}: {header.get('resultMsg')}")
        return data["response"]["body"]["items"]["item"]
