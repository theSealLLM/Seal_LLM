"""scripts/e2e 공통: src 경로 등록."""

from __future__ import annotations

import sys
from pathlib import Path

# pyproject가 package = false라서 theseal 패키지를 설치하지 않는다. src를 직접 경로에 넣는다.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 콘솔 코드페이지가 한글을 못 받는 환경(cp1252 등)에서 print가 죽지 않도록
sys.stdout.reconfigure(errors="replace")
