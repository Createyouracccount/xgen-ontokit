"""감정 하네스 회귀 — 누설 차단과 자 적용이 **구조적으로** 보장되는지 건다.

R4·R5·R6·R7 네 라운드가 전부 감정 설계에서 죽었다. 이 테스트가 지키는 것은
처치가 아니라 **측정 절차**다.
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "harness"))

from packet import ALLOWED, LeakError, build, write          # noqa: E402
from ruler import apply as ruler_apply                        # noqa: E402
from ruler import is_facility                                 # noqa: E402


def _entries(n=6):
    return [{"label": f"L{i}", "class": "기관", "contexts": [f"ctx{i}"],
             "group": "A" if i % 2 else "B"} for i in range(n)]


def test_packet_only_whitelisted_fields():
    """배포본에는 화이트리스트 필드만 나간다 — 정답을 넣을 방법이 없다."""
    pack, key = build(_entries(), seed=7, note="시험")
    for it in pack["items"]:
        assert set(it) <= set(ALLOWED), it
    # 정답은 별도 대조표에만 있다
    assert all("group" in k for k in key["key"])
    assert not any("group" in it for it in pack["items"])


def test_packet_id_assigned_after_shuffle():
    """id ↔ 원 집합 상관이 **구조적으로** 0 이다.

    R7 에서 id 를 셔플 전에 부여해 `id 0~123 = 소멸` 로 갈렸고, 심판 A 가
    id 오름차순으로 판정했다(Spearman +1.0). 그 경로를 막는다.
    """
    entries = [{"label": f"L{i}", "class": "기관", "contexts": ["c"],
                "group": "소" if i < 20 else "신"} for i in range(40)]
    pack, key = build(entries, seed=11)
    g = {k["id"]: k["group"] for k in key["key"]}
    first20 = [g[i] for i in range(20)]
    # 셔플 후 id 부여이므로 앞쪽 20개가 한 집합으로 몰릴 수 없다
    assert len(set(first20)) == 2, first20


def test_packet_rejects_answer_in_note():
    """note·지침으로 새는 경로도 직렬화 후 검사로 막는다."""
    with pytest.raises(LeakError):
        build(_entries(), seed=1, note="소멸 집합 감정")


def test_packet_rejects_same_directory():
    """배포본과 정답표가 같은 디렉터리면 거부한다 — 심판이 탐색하면 도달한다."""
    pack, key = build(_entries(), seed=1)
    d = tempfile.mkdtemp()
    with pytest.raises(LeakError):
        write(pack, key, os.path.join(d, "p.json"), os.path.join(d, "k.json"))
    d2 = tempfile.mkdtemp()
    r = write(pack, key, os.path.join(d, "p.json"), os.path.join(d2, "k.json"))
    assert r["n"] == 6


def test_ruler_catches_facilities():
    """사전 등록 자 — 시설은 3분류 어디에도 속하지 않는다(거짓)."""
    for lb in ("서울추모공원", "제2여객터미널", "e편한세상부평역센트럴파크",
               "원당역롯데캐슬스카이엘", "시안가족추모공원", "제1전시관",
               "인천두산위브더센트럴", "권곡동 아산한신더휴"):
        assert is_facility(lb), lb


def test_ruler_does_not_overreach():
    """정상 기관·지역을 잡으면 안 된다.

    ⚠️ 실측 오탐: `센터` 를 접미 목록에 넣었더니 `한국유통정보센터`(기관)가 잡혔다.
       과대 규칙은 R4·R5 를 죽인 유형이라 좁게 유지한다.
    """
    for lb in ("산업통상자원부", "CJ올리브네트웍스", "서울", "한국전력공사",
               "코스피", "한국유통정보센터", "한국연구재단"):
        assert not is_facility(lb), lb


def test_ruler_overrides_majority():
    """심판 다수결이 참이어도 사전 등록 자가 덮어쓴다(K-1).

    R5 에서 심판 3인이 **만장일치로** 아파트 단지명을 참으로 봤고, R7 에서 같은
    유형이 이득 51건 중 6건 들어왔다. 산문이 아니라 코드가 처리한다.
    """
    v = {0: "참", 1: "참", 2: "거짓", 3: "판정불가"}
    lb = {0: "서울추모공원", 1: "산업통상자원부", 2: "웍스", 3: "제2여객터미널"}
    r = ruler_apply(v, lb)
    assert r["판정"][0] == "거짓"          # 시설 → 덮어씀
    assert r["판정"][1] == "참"            # 기관 → 그대로
    assert r["판정"][2] == "거짓"
    assert r["판정"][3] == "판정불가"      # 참이 아니면 덮어쓰지 않는다
    assert r["덮어쓴수"] == 1
    assert r["자_덮어씀"][0]["label"] == "서울추모공원"


def test_packet_context_not_scanned_for_forbidden_words():
    """원문 컨텍스트는 금지어 검사에서 제외된다 — 오탐으로 정당한 패킷이 막히면 안 된다.

    ⚠️ 실측: 뉴스 원문에 `조직을 신설한 데 이어` · `처치실` 이 자연히 나와
       8건이 오탐으로 걸렸다. 누설은 구조 필드로 일어나지 원문 인용으로 일어나지 않는다.
    """
    e = [{"label": "LG유플러스", "class": "기관",
          "contexts": ["데이터 분석 전담 조직을 신설한 데 이어 처치실을 두고 있다"],
          "group": "G1"}]
    pack, _ = build(e, seed=3, note="라벨 감정")      # 예외 없이 생성돼야 한다
    assert pack["n"] == 1
    assert "신설" in pack["items"][0]["contexts"][0]  # 원문은 그대로 보존

    # 그러나 구조 필드로 새면 여전히 막힌다
    with pytest.raises(LeakError):
        build(e, seed=3, note="소멸 집합")
    with pytest.raises(LeakError):
        build([{**e[0], "label": "_set"}], seed=3)


def test_packet_injects_coexistence_warning():
    """집합 감정 패킷에 **공존 경고가 강제 주입**되는가 — 호출자가 뺄 수 없다.

    ⛔ R8 에서 심판 3인이 만장일치로 "온전형과 조각이 동시 등재된다"고 지적했고
       차기 표적이 됐으나 **실측 결과 틀렸다** — 패킷 내 부분문자열 쌍 108종 중
       103종(95%)이 서로 다른 집합이라 한 빌드에 공존한 적이 없다.
       심판이 before/after 섞인 패킷을 "현재 상태"로 오독한 것이다.
    """
    from packet import COEXIST_WARNING

    pack, _ = build(_entries(), seed=5, note="라벨 감정")
    assert pack["판정지침"]["공존_경고"] == COEXIST_WARNING
    assert "공존한다고" in pack["판정지침"]["공존_경고"]

    # 호출자가 지침을 통째로 주더라도 경고는 살아남는다
    pack2, _ = build(_entries(), seed=5, instructions={"값": ["참", "거짓"]})
    assert "공존_경고" in pack2["판정지침"]
    assert pack2["판정지침"]["값"] == ["참", "거짓"]


def test_context_prefers_standalone_occurrence():
    """근거 컨텍스트는 **독립 출현을 우선**한다 — R7 지적·R10 재발.

    구 생성기는 첫 출현을 잡아 짧은 라벨이 다른 단어 내부에 걸렸다.
    실측: `서부` → 심판이 본 것은 `발굴에서부터`·`단계에서부터` 뿐이고
    실제 개체 자리 `남동·동서·남부·서부·중부발전` 은 못 봤다. R10 에서 14/152=9.2%.
    """
    from context import contexts

    chunks = [
        {"chunk_text": "후보물질 발굴에서부터 임상 허가까지 진행한다."},
        {"chunk_text": "한국전력과 남동·동서·남부·서부·중부발전 등 5개 자회사가 있다."},
    ]
    got = contexts(["서부"], chunks)["서부"]
    # 독립 출현(`·서부·`)이 선택돼야 한다
    assert any("중부발전" in c for c in got), got
    assert not any("발굴에서부터" in c for c in got), got

    # 독립 출현이 없으면 [내부출현] 표시를 달아 넘긴다 — 조각이라는 정보다
    got2 = contexts(["부터"], chunks)["부터"]
    assert all(c.startswith("[내부출현]") for c in got2), got2

    # 원문에 없으면 그 사실을 알린다
    got3 = contexts(["없는문자열"], chunks)["없는문자열"]
    assert "찾지 못함" in got3[0]
