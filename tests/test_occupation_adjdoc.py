"""adjdoc 증거 모드(id4, 도메인 코퍼스 opt-in) — 긴 라벨 doc 약증거 게이트.

판례: 브렛 테일러(트위터 이사회 의장, 한글 5음절)가 어휘집 동명 배우와 충돌해
무증거 타이핑되던 장라벨 동명이인 오탐(news2k 실측 1/25)을 차단한다.
"""
from ontokit.instance_typing.occupation import _evidence_ok


TEXTS = [
    "브렛 테일러 트위터 이사회 의장은 계약 이행을 강제하기 위한 소송을 제기하겠다고 밝혔다.",
    "배우 전지현이 새 드라마에 출연한다.",
    "가수 수지가 신곡을 발표했다.",
]


def test_adjdoc_blocks_long_label_without_cue():
    # 긴 라벨 + 출현 문서에 직업 단서 부재 → 차단
    assert not _evidence_ok("브렛 테일러", "배우", TEXTS, "adjdoc")


def test_adjdoc_keeps_long_label_with_cue():
    texts = ["배우 크리스토퍼 놀란상을 받은 티모테 샬라메가 내한했다."]
    assert _evidence_ok("티모테 샬라메", "배우", texts, "adjdoc")
    assert not _evidence_ok("티모테 샬라메", "화가", texts, "adjdoc")


def test_adjdoc_short_label_uses_adj():
    # 짧은 라벨은 기존 adj 인접 게이트 그대로 (동결분 무변경)
    assert _evidence_ok("수지", "가수", TEXTS, "adjdoc")
    assert not _evidence_ok("수지", "화가", TEXTS, "adjdoc")


def test_existing_modes_unchanged():
    # adj(기본): 긴 라벨 면제 유지
    assert _evidence_ok("브렛 테일러", "배우", TEXTS, "adj")
    assert _evidence_ok("브렛 테일러", "배우", TEXTS, "off")
