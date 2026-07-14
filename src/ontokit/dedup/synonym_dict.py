"""사전 기반 동의어 dedup 채널 — 우리말샘 비슷한말로 의미변이 병합. LLM 0회.

형태소 정규화(DeterministicDedup)는 표면변이(삼성전자=삼성 전자)만 병합. 의미변이
(전자상거래=이커머스, 컴퓨터=전산기)는 표면 문자열로 원리적 불가 — 외부 편찬 사전
필요. 외부 gold 심판루프(위키 redirect)로 검증(2026-07-14, 게이트 0.80 경계).
정본 docs/ontokit_ER_심판_텍스트접근_채널교체_2026_07_14.md.

⚠️ ER 축 실증 결론(심판 5R): 임베딩은 주제근접≠동의어 원리적 미분리(AUC 0.81 천장),
전통사전은 현대 고유명·신조어 미수록(도메인 미스매치). 사전 채널은 **고정밀 확정
동의어 병합**(P 1.000)이 역할이지 광범위 recall 아님. 신중한 opt-in.

자원: 우리말샘(국립국어원, CC BY-SA 2.0 KR, 상업 OK). 비슷한말 union-find 스냅샷을
빌드타임에 만들어(eval/entity_resolution/er_urimalsam.py) 경량 TSV(표면형→집합대표)로
로드. env ONTOKIT_SYNONYM_DICT=경로 로 켠다. 미지정 시 이 채널 미생성(형태소만).

버그수정(심판 R4): 비슷한말 비이행성 — union-find sense 보존 + 거대 블롭 폐기
(미국=일본=한국 오병합 방지). TSV 는 그 수정본 산출물.
"""
from __future__ import annotations
import os
import re

_HOMONYM = re.compile(r"\d+$")
_CLEAN = re.compile(r"[\s_\-·•/\\()（）]+")


def _surface(w: str) -> str:
    """동형번호·공백·기호 제거 표면형(조회 키)."""
    return _HOMONYM.sub("", _CLEAN.sub("", (w or "").strip()))


class SynonymDictDedup:
    """우리말샘 비슷한말 스냅샷(표면형→집합대표들). 같은 대표 공유 = 동의어.

    env ONTOKIT_SYNONYM_DICT=<tsv> 또는 tsv_path 인자로 스냅샷 지정.
    TSV 형식(eval/er_urimalsam._save_cache 산출): "표면형\\t대표1 대표2 ...".
    """

    ENV_PATH = "ONTOKIT_SYNONYM_DICT"

    def __init__(self, tsv_path: str | None = None):
        path = tsv_path or os.getenv(self.ENV_PATH)
        if not path:
            raise ValueError(
                f"동의어 사전 스냅샷 경로 미지정 — {self.ENV_PATH} env 또는 tsv_path 필요. "
                "생성: eval/entity_resolution/er_urimalsam.py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"동의어 사전 스냅샷 없음: {path}")
        self._s2reps: dict[str, set] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2 and parts[1]:
                    self._s2reps[parts[0]] = set(parts[1].split())

    def are_synonyms(self, a: str, b: str) -> bool:
        ra = self._s2reps.get(_surface(a))
        rb = self._s2reps.get(_surface(b))
        return bool(ra and rb and (ra & rb))

    def rename_by_synonym(self, names: list[str]) -> dict[str, str]:
        """같은 사전 동의어 집합의 표기들 → 첫 등장을 canonical 로. {중복: canonical}.

        형태소 dedup 과 병합용 — DeterministicDedup 이 못 잡는 의미변이만 추가.
        """
        rep2canon: dict[frozenset, str] = {}
        rename: dict[str, str] = {}
        for name in names:
            if not name:
                continue
            reps = self._s2reps.get(_surface(name))
            if not reps:
                continue
            key = frozenset(reps)
            if key in rep2canon:
                if name != rep2canon[key]:
                    rename[name] = rep2canon[key]
            else:
                rep2canon[key] = name
        return rename

    def size(self) -> int:
        return len(self._s2reps)
