"""0729 코드 감사 수리분 회귀 잠금 (docs/ontokit/ontokit_코드감사_파라미터소실_이중경로_2026_07_29.md).

세 수리 모두 "기존 채택분이 한쪽 경로에만 적용돼 있던" 유형이라, 잠그지 않으면
같은 형태로 재발한다. 각 테스트는 감사에서 실제 재현한 케이스를 그대로 쓴다.
"""
import pytest

from ontokit.filter.class_promotion import ClassPromotionFilter
from ontokit.instance_typing.occupation import _evidence_ok
from ontokit.search.improvements import blend_score


# ── C1: partition() 이 person_dom(R15 인명 게이트)을 전달하는가 ──────────
# 감사 시점 결함: decide() 는 5개 signal 을 받는데 partition() 이 person_dom 만
# 누락 → batch 경로에서 R15 게이트 미발화(인명이 owl:Class 로 승격).
class TestPersonDomForwarding:
    def test_partition_forwards_person_dom(self):
        f = ClassPromotionFilter()
        kept, dropped = f.partition(
            [{"name": "소크라테스", "df": 5, "has_rel": True, "person_dom": True}])
        assert kept == []
        assert [r for _, r in dropped] == ["person"]

    def test_partition_matches_decide(self):
        """동일 입력에서 partition 과 decide 의 판정이 일치해야 한다."""
        f = ClassPromotionFilter()
        item = {"name": "소크라테스", "df": 5, "has_rel": True, "person_dom": True}
        d = f.decide(item["name"], df=item["df"], has_rel=item["has_rel"],
                     person_dom=item["person_dom"])
        kept, _ = f.partition([item])
        assert bool(kept) is d.keep

    def test_person_dom_absent_keeps_legacy_behavior(self):
        """키 미지정 = 게이트 미발화(기존 동작 보존)."""
        f = ClassPromotionFilter()
        kept, dropped = f.partition([{"name": "소크라테스", "df": 5, "has_rel": True}])
        assert [i["name"] for i in kept] == ["소크라테스"]
        assert dropped == []


# ── C2: doc 모드에도 한글 경계 매칭이 적용되는가 ────────────────────────
# 감사 시점 결함: B1/B7 심판 수리(경계 정규식)가 adj·adjdoc 에만 들어가고
# doc 분기는 맨 `in` 검사 → '무역수지'의 '수지', '가수왕'의 '가수' 관통.
FP_TEXT = ["무역수지가 개선되었고 가수왕 선발"]          # 인물 무관 — 차단돼야 함
TP_ADJ = ["가수 수지가 신곡을 발표했다"]                  # 인접 정탐
TP_FAR = ["수지는 활동을 시작했다. 그는 가수다."]         # 원거리 정탐(doc 존재 이유)


class TestDocModeBoundary:
    @pytest.mark.parametrize("mode", ["doc", "adj", "adjdoc"])
    def test_substring_leak_blocked_in_all_modes(self, mode):
        assert _evidence_ok("수지", "가수", FP_TEXT, mode) is False

    @pytest.mark.parametrize("mode", ["doc", "adj"])
    def test_adjacent_true_positive_survives(self, mode):
        assert _evidence_ok("수지", "가수", TP_ADJ, mode) is True

    def test_doc_keeps_long_range_recall(self):
        """doc 의 '약한 증거' 성격 유지 — 인접 창 없이 동일 청크 공기만 요구."""
        assert _evidence_ok("수지", "가수", TP_FAR, "doc") is True

    def test_off_mode_unchanged(self):
        assert _evidence_ok("수지", "가수", FP_TEXT, "off") is True


# ── C4: blend_score 의 bool 오인식 + vrng==0 역전 ───────────────────────
# 감사 시점 결함 2건: (a) bool ⊂ int 라 sentinel True 가 vscore 1.0 으로 둔갑해
# 날조 점수 0.7 생성 (b) vrng==0 에서 vscore 있는 청크(0.24)가 없는 청크(0.80)
# 아래로 역전.
class TestBlendScore:
    @pytest.mark.parametrize("sentinel", [True, False])
    def test_bool_not_treated_as_score(self, sentinel):
        """bool 은 수치가 아니라 결측으로 취급 — 날조 confidence 금지."""
        assert blend_score(sentinel, knorm=0.0, vmin=0.0, vrng=1.0) == 0.0

    def test_no_inversion_when_vrng_zero(self):
        """정규화 불가 구간에서 vscore 보유 청크가 결측 청크보다 낮으면 안 된다."""
        has = blend_score(0.9, knorm=0.8, vmin=0.9, vrng=0.0)
        missing = blend_score(None, knorm=0.8, vmin=0.9, vrng=0.0)
        assert has == pytest.approx(missing)

    def test_normal_path_unchanged(self):
        """정상 정규화 경로는 수리 전과 동일해야 한다."""
        assert blend_score(0.9, knorm=0.5, vmin=0.5, vrng=1.0) == pytest.approx(
            0.7 * 0.4 + 0.3 * 0.5)

    def test_missing_vscore_floor_still_works(self):
        """원래 의도(결측 → knorm floor)는 보존."""
        assert blend_score(None, knorm=0.6, vmin=0.0, vrng=1.0) == pytest.approx(0.6)
