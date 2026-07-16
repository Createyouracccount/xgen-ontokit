"""인스턴스 타이핑 품질 모듈 (R11) — 단일 진실원천.

독립 타이핑 패스(재빌드 없는 루프)와 정기 빌드가 **같은 함수**를 호출한다
(별도 구현 금지 — 라이프사이클 심판 요건). 전부 결정론·LLM 0콜.
"""
from ontokit.instance_typing.hygiene import label_ok

__all__ = ["label_ok"]
