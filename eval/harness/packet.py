"""감정 패킷 생성기 — **정답 누설을 코드로 물리 차단**한다.

## 왜 필요한가

R4·R5·R6·R7 네 라운드가 **전부 감정 설계에서** 죽었다. 처치 기제는 매번 건전했다.
R7 에서 드러난 누설 두 가지:

1. **id 구간 누설** — id 를 셔플 **전에** 부여해서 `id 0~123 = 소멸 · 124~222 = 신설` 로
   완전히 갈렸다. 심판 A 는 실제로 id 오름차순으로 판정했고(Spearman **+1.0**,
   B·C 는 −0.087) 세 심판 중 **처치에 가장 유리했다**.
2. **정답 산출물 방치** — 심판 작업 디렉터리에 `_set` 필드를 가진 원본과
   `r7_ab_result.json`(소멸·신설 전체 목록)이 그대로 있었다.

산문 규율("맹검을 지킨다")로는 막히지 않았다. **구조로 막는다.**

## 이 모듈이 강제하는 것

- 심판 배포본은 **화이트리스트 필드만** 나간다(`id·label·class·contexts`).
  집합·처치 방향·설계자 판정은 **넣을 방법이 없다** — 통과 필드를 상수로 고정했다.
- **id 는 셔플 뒤에 부여**한다. 따라서 id 와 원 집합의 상관은 구조적으로 0 이다.
- 정답 대조표(`answer_key`)는 **별도 파일**로 나가고, 배포본과 **다른 디렉터리**를
  강제한다. 같은 디렉터리면 예외를 던진다.
- 배포본에 금지 문자열이 남아 있는지 **직렬화 후 검사**한다. 남아 있으면 예외.
"""
from __future__ import annotations

import json
import os
import random

# 심판 배포본에 나갈 수 있는 **유일한** 필드. 여기 없는 것은 나가지 않는다.
ALLOWED = ("id", "label", "class", "contexts")

# 배포본에 있으면 안 되는 문자열.
#
# ⚠️ **원문 컨텍스트는 검사 대상에서 뺀다.** 뉴스 원문에는 `조직을 신설한 데 이어` ·
#    `처치실` 같은 어휘가 자연히 나온다(실측 8건). 컨텍스트까지 훑으면 오탐으로
#    정당한 패킷 생성이 막힌다. 누설은 **구조 필드**(note·지침·필드명)로 일어나지
#    원문 인용으로 일어나지 않는다 — 검사 범위를 거기로 좁힌다.
FORBIDDEN = ("_set", "소멸", "신설", "before", "after", "처치", "verdict", "판정_설계자")


class LeakError(RuntimeError):
    """정답이 배포본에 새어나갈 수 있는 상태 — 생성 자체를 거부한다."""


# 집합 감정 패킷에 **반드시** 들어가야 하는 경고.
#
# ⛔ R8 에서 심판 3인이 **만장일치로** "온전형과 조각이 동시에 로스터에 등재된다"고
#    지적했고, 그것이 차기 표적으로 지정됐다. **실측 결과 틀렸다** — 패킷 내 부분문자열
#    쌍 108종 중 **103종(95%)이 서로 다른 집합**(처치 전/후)이라 한 빌드에 공존한 적이
#    없다. 심판이 before/after 를 섞은 패킷을 **"로스터의 현재 상태"로 오독**한 것이다.
#
#    맹검을 위해 섞는 것은 옳다. 그러나 **섞였다는 사실 자체는 알려야** 한다.
#    안 알리면 심판의 패턴 관측이 통째로 오염된다. 이 경고를 지침에 강제 주입한다.
COEXIST_WARNING = (
    "⚠️ 이 항목들은 **서로 다른 시점의 산출물이 섞여** 있다. 두 항목이 같은 빌드에 "
    "공존한다고 **가정하지 말 것**. 항목 간 관계(중복·포함·쌍)를 근거로 판정하지 말고, "
    "각 항목을 **원문 컨텍스트만 보고 독립적으로** 판정하라."
)


def build(entries, seed, note="", instructions=None):
    """맹검 패킷 + 정답 대조표를 만든다.

    entries: [{"label":…, "class":…, "contexts":[…], "group":…}, …]
             `group` 이 정답(어느 집합인지). **배포본에는 절대 안 들어간다.**
    반환: (blind_pack, answer_key)
    """
    items = [dict(e) for e in entries]
    random.Random(seed).shuffle(items)          # ① 셔플 먼저

    blind, key = [], []
    for new_id, it in enumerate(items):         # ② id 는 셔플 **뒤에** 부여
        row = {k: it[k] for k in ALLOWED if k in it}
        row["id"] = new_id
        extra = set(row) - set(ALLOWED)
        if extra:
            raise LeakError(f"허용되지 않은 필드가 배포본에 있다: {sorted(extra)}")
        blind.append(row)
        key.append({"id": new_id, "label": it.get("label"), "group": it.get("group")})

    inst = dict(instructions or {})
    inst["공존_경고"] = COEXIST_WARNING        # 강제 주입 — 호출자가 뺄 수 없다
    pack = {"seed": seed, "n": len(blind), "note": note,
            "판정지침": inst, "items": blind}

    # ③ 금지 문자열 실검사 — **컨텍스트를 제외한** 구조 부분만 본다.
    #    note·지침·필드명·라벨로 새는 경로를 막는다. 원문 인용은 검사하지 않는다(위 주석).
    probe = {k: v for k, v in pack.items() if k != "items"}
    probe["items"] = [{k: v for k, v in it.items() if k != "contexts"}
                      for it in pack["items"]]
    dumped = json.dumps(probe, ensure_ascii=False)
    hit = [w for w in FORBIDDEN if w in dumped]
    if hit:
        raise LeakError(f"배포본에 정답 단서가 남아 있다: {hit}")

    return pack, {"seed": seed, "n": len(key), "key": key}


def write(pack, key, pack_path, key_path):
    """배포본과 정답표를 **다른 디렉터리**에 쓴다. 같으면 거부한다."""
    pd = os.path.dirname(os.path.abspath(pack_path))
    kd = os.path.dirname(os.path.abspath(key_path))
    if pd == kd:
        raise LeakError(
            f"배포본과 정답표가 같은 디렉터리다({pd}). 심판이 탐색하면 정답에 도달한다.")
    for path, obj in ((pack_path, pack), (key_path, key)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    return {"pack": pack_path, "key": key_path, "n": pack["n"]}
