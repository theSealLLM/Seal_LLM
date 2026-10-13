"""기상청 날씨 툴 (기상청 API허브 · 단기예보 조회서비스). 지금 날씨와 오늘~모레 예보를 함께 돌려준다.

- API: https://apihub.kma.go.kr  /api/typ02/openApi/VilageFcstInfoService_2.0
    getUltraSrtNcst  초단기실황 (지금 날씨: 기온 · 습도 · 1시간 강수량 · 강수 형태 · 풍속)
    getVilageFcst    단기예보 (오늘~모레: 최저 · 최고 기온, 하늘 상태, 강수 형태, 강수 확률)
- 인증키: .env의 KMA_AUTH_KEY (없으면 레지스트리가 이 툴을 등록하지 않는다)
- 지역: 아래 REGIONS 표(17개 시 · 도 + 주요 도시)만 지원한다. 위경도를 기상청 격자(nx, ny)로 바꿔 조회한다.
  표에 없는 지역은 ValueError → tool_caller가 실패로 기록하고, 프롬프트 지시에 따라 검색 툴로 넘어간다
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from langchain_core.tools import tool

from ...config import get_config

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
WEEKDAYS_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
FCST_ISSUE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)  # 단기예보 발표 시각
NCST_READY_MIN = 40  # 초단기실황은 정시 자료를 이 분 이후에 조회한다 (여유 포함)
FCST_READY_MIN = 15  # 단기예보는 발표 후 이 분 이후에 조회한다 (여유 포함)
HTTP_TIMEOUT_SEC = 8


# ---------------------------------------------------------------- 지역 · 격자

def resolve_region(location: str) -> str:
    """'서울특별시', '부산 해운대구', '경기도' → 표의 지역 이름. 없으면 ValueError."""
    text = (location or "").strip()
    candidates = [text.replace(" ", "")] + text.split()
    for cand in candidates:
        cand = ALIASES.get(cand, cand)
        if cand in REGIONS:
            return cand
        for suffix in SUFFIXES:
            if cand.endswith(suffix) and cand[: -len(suffix)] in REGIONS:
                return cand[: -len(suffix)]
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
    return datetime.now(ZoneInfo(get_config().timezone))


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


# ---------------------------------------------------------------- 호출 · 해석

def _call(operation: str, base_date: str, base_time: str, nx: int, ny: int) -> list[dict]:
    config = get_config()
    params = {"pageNo": 1, "numOfRows": 1500, "dataType": "JSON", "base_date": base_date,
              "base_time": base_time, "nx": nx, "ny": ny, "authKey": config.kma_auth_key}
    resp = httpx.get(f"{config.kma_api_base}/{operation}", params=params, timeout=HTTP_TIMEOUT_SEC)
    data = resp.json()
    if "result" in data:  # API허브 자체 오류 (인증키 등)
        raise RuntimeError(f"기상청 API 오류 {data['result'].get('status')}: {data['result'].get('message')}")
    header = data["response"]["header"]
    if header.get("resultCode") != "00":
        raise RuntimeError(f"기상청 API 오류 {header.get('resultCode')}: {header.get('resultMsg')}")
    return data["response"]["body"]["items"]["item"]


def summarize_now(items: list[dict]) -> dict:
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


def summarize_forecast(items: list[dict], today: datetime) -> list[dict]:
    """날짜별(오늘 · 내일 · 모레) 최저 · 최고 기온과 오전 · 오후의 하늘 · 강수 · 강수 확률."""
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


def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value  # "강수없음" 같은 문자열은 그대로
    return int(f) if f.is_integer() else f


@tool
def get_weather(location: str) -> dict:
    """기상청 자료로 지금 날씨와 오늘 · 내일 · 모레 예보를 알려 준다. 날씨 · 기온 · 비 · 눈 질문에 먼저 쓴다.
    location은 시 · 도 또는 주요 도시 이름 (예: 서울, 부산, 제주, 수원). 지역을 말하지 않았으면 '서울'.
    am은 06~12시, pm은 12~18시. pop_max_pct는 강수 확률 최댓값(%)."""
    region = resolve_region(location)
    nx, ny = latlon_to_grid(*REGIONS[region])
    now = now_kst()
    result = {"location": region, "grid": {"nx": nx, "ny": ny}, "source": "기상청 단기예보"}
    result["now"] = summarize_now(_call("getUltraSrtNcst", *ncst_base(now), nx, ny))
    base_date, base_time = fcst_base(now)
    result["forecast_issued"] = f"{base_date} {base_time}"
    result["forecast"] = summarize_forecast(_call("getVilageFcst", base_date, base_time, nx, ny), now)
    return result
