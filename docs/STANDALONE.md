# 독립 실행 — XGEN·도커·서버 없이 문서 → 그래프 → 질의 → 병합

[← README 로 돌아가기](../README.md)

측정 하네스에서 검증된 경로([`harness/docs/L02`](../harness/docs/L02_루프탈출_판정.md))를 `pip install` 한 라이브러리만으로 쓴다.
LLM 은 질문을 질의계획으로 바꿀 때만 쓰고(선택), 그래프 생성·질의·병합에는 쓰지 않는다.
옮긴 코드가 측정 때와 같은 답을 낸다는 증명은 [`harness/docs/S01`](../harness/docs/S01_라이브러리승격_동등성.md).

## 1. 설치

```
pip install "xgen-ontokit[korean,ner,owl] @ git+https://github.com/Createyouracccount/xgen-ontokit@<태그>"
```

| extra | 무엇을 위해 |
|---|---|
| `owl` (rdflib) | 그래프 투영(`ontokit.graph`)·인메모리 질의(`ontokit.query.run`)·`ontokit query` |
| `korean`, `ner` (Kiwi, transformers·torch) | `ontokit build`(추출). KoELECTRA NER 가중치는 첫 실행에 Hugging Face 에서 받는다 |
| `oxigraph` (pyoxigraph) | 디스크에 남는 내장 SPARQL 저장소(서버·JVM 없음) — 선택 |

⚠️ **추출 결과는 kiwipiepy 버전에 따라 달라진다.** L02 측정 그래프는 kiwipiepy 0.23.2 로 만들었다. 0.24.0 에서는 같은 60문서에서
개체 902→905, 하위 클래스 32→26 이 됐다(어느 쪽이 나은지는 측정하지 않았다). 측정 그래프를 재현하려면
`pip install "kiwipiepy==0.23.2"` 로 고정한다. torch·transformers 버전은 관계 점수 소수 넷째 자리만 바꿨고 그래프는 같았다(S01 §7).

## 2. 명령줄

```
ontokit build docs.jsonl -o graph.ttl          # LLM 0회. 폴더(.txt·.md)도 받는다
ontokit query graph.ttl '{"op":"list","return":"x","where":[{"t":"isa","v":"x","class":"가수"}]}'
ontokit ask graph.ttl "이 문서 모음의 가수를 모두 나열해줘"   # 계획만 LLM — 아래 env 필요
```

- 입력 `.jsonl`: 한 줄 = 한 청크 `{"doc_id", "text", "title"?, "chunk_index"?}`. 폴더 입력은 파일마다 문단 경계로
  1,200자(NER 이 한 번에 보는 길이) 이하 청크로 나누고, `doc_id` = 상대 경로, `title` = 파일 이름.
- `query` 결과: 노드마다 `name`·`names`(별칭)·`uri`·`docs`(그 노드가 표제이거나 언급된 문서 `doc_id`).
- `ask` 는 OpenAI 호환 엔드포인트를 쓴다: `ONTOKIT_LLM_URL`(…`/v1` 까지)·`ONTOKIT_LLM_MODEL`, 선택 `ONTOKIT_LLM_API_KEY`.
  설정이 없거나 LLM 출력이 계획 형식이 아니면 **실패로 멈춘다**(빈 결과로 바꾸지 않는다).

## 3. 파이썬 — 이미 벡터 RAG 가 있을 때

검증된 처치는 **두 단계**다. ① 판독기 프롬프트의 청크 뒤에 그래프 결과 블록을 넣고 ② 그 판독기의 답에 그래프 결과를 합친다.
(블록 없이 ② 만 쓰는 조합은 측정되지 않았다 — S01 §3.)

```python
from rdflib import Graph
from ontokit import planner, query, merge

g = Graph().parse("graph.ttl")                       # 또는 ontokit.graph.project(raw, docs)[0]
plan = planner.parse(my_llm(planner.PROMPT + question))   # 계획은 직접 써도 된다
nodes = query.run(g, plan)["nodes"]                   # [{"uri", "names"}]

prompt = f"{chunks_text}\n\n{merge.graph_block([n['names'][0] for n in nodes])}질문: {question}"
reader = my_reader_llm(prompt)                        # {"answers": [...], "count"?: n}
answer = merge.merge(reader["answers"], nodes, op=plan["op"], reader_count=reader.get("count"))
# answer["items"] = 판독 답 + 그래프 노드(중복 제거 안 함), answer["count"] = op 가 count 일 때 그래프 노드 수(없으면 판독 개수)
# 측정에서는 op 를 계획이 아니라 문항 형태(개수 질문인가)로 정했다 — 계획기가 op 를 틀리면 둘이 갈린다
```

질의계획 형식과 관계 어휘(KLUE-RE 라벨 12종)는 `ontokit.planner.PROMPT` 와 `ontokit.graph.RELATIONS` 에 있다.

## 4. 저장소 — 서버 없이, 여러 백엔드

| 백엔드 | 형태 | 같은 답 증명 |
|---|---|---|
| 인메모리 `query.run(g, plan)` | rdflib, 프로세스 안 | Fuseki 와 EVAL 4개 그래프 × 계획 2종 문항별 전부 동일(S01 §5) |
| `ontokit.backends.oxigraph.OxigraphStore(path)` | 내장 SPARQL, 디렉터리 저장 | 인메모리와 문항별 동일(S01 §6) |
| 원격 SPARQL 저장소 `await query.run_plan(store, plan, graph)` | `async sparql_query(q)` 를 가진 아무 객체 | L02 측정(Fuseki) 그대로 |

```python
from ontokit.backends.oxigraph import OxigraphStore
st = OxigraphStore("graphs/")            # 디렉터리 저장. 읽기만 하는 프로세스는 OxigraphStore("graphs/", read_only=True)
st.load(g, "urn:my-graph")               # 같은 이름은 교체. 트리플 수가 어긋나면 예외
nodes = (await query.run_plan(st, plan, "urn:my-graph"))["nodes"]
```

ontokit 은 Fuseki·graphstore 에 의존하지 않는다. 측정 하네스도 Oxigraph 로 옮겼다 — 같은 측정 그래프 13개(141만 트리플)가
Fuseki 에서 3.0GB, Oxigraph 에서 약 0.4GB 였고, 지우고 다시 싣기를 반복해도 쌓이지 않았다(S01 §6).
Cypher(Neo4j 등 LPG)로 같은 계획을 실행하는 경로는 아직 하네스(`harness/lpg.py`, graphstore 경유)에만 있다.

## 5. 관계 채널 — 가중치가 있어야 켜진다

관계(출생지·소속·위치 등)는 로컬 관계 인코더(KLUE-RE 파인튜닝, 261MB)가 있어야 나온다. 이 가중치는 아직 배포하지 않는다
(공개 여부·라이선스 확인 중). 가중치가 없으면 `ontokit build` 는 그렇게 알리고 **관계 0건, 타입·클래스만** 만든다 —
이 상태에서는 `isa` 계획만 답이 나온다.

L02 측정 그래프의 추출 설정(가중치가 있을 때):

```
ONTOKIT_RELATION_ENCODER_MODEL=<가중치 경로>
ONTOKIT_RE_MAX_PAIRS_PER_SENT=30  ONTOKIT_RE_TOPIC_SUBJECT=1  ONTOKIT_LOC_REL=1  ONTOKIT_RE_SUFFIX_ORG=1
```

## 6. 한계 — 측정되지 않았거나 맞지 않는 곳

- **문서 = 개체인 자료에 맞춘 질의 규칙.** 변수 `x` 는 "문서 제목과 이름이 같은 개체"(표제 개체)로 한정된다. 위키처럼
  문서마다 주제 개체가 하나인 자료에서 측정했다. 제목이 개체 이름이 아닌 문서(보고서·뉴스·상품 설명)에서는 `x` 로 묻는
  목록이 비기 쉽다. 상수로 시작하는 계획(`{"const": "ㄱ회사"}` 의 설립자 등)은 이 제한을 받지 않는다.
- **도메인·판독기·정답**: 한국어 위키 리드, 로컬 Qwen3-8B 판독기, Wikidata 정답(사람 감사 없음)에서만 검증됐다.
- **이득이 난 질문**: 열거·관계·다중 홉. 집계(개수)·교차 조건은 이득이 없었다(개수 규칙의 EVAL 기여 0 — S01 §2).
- **크기**: 그래프는 정답 그래프의 약 1/10 효과다(추출 재현율이 상한).
