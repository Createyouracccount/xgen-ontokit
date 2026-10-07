"""하네스 공통 LLM 호출 — 백엔드를 env 로 고른다(코드 경로는 하나).

  HARNESS_LLM=ollama (기본)            : OLLAMA_URL(/api/generate), 모델 HARNESS_LLM_MODEL
  HARNESS_LLM=openai                   : HARNESS_LLM_URL(/v1/chat/completions) — 로컬 MLX 서버·DGX vLLM 공통
                                          모델 HARNESS_LLM_MODEL

temperature 0 · seed 0 고정. Qwen3 사고 모드는 끈다(ollama: think=false / openai: /no_think 소프트 스위치).
JSON 은 잘린 꼬리를 살리는 복구 파서로 읽고, 복구 불가면 예외(0건으로 조용히 넘기지 않는다).
"""
import json
import os
import urllib.request


def backend():
    return os.getenv("HARNESS_LLM", "ollama")


def generate_json(prompt, *, model=None, num_ctx=4096, max_tokens=1024, timeout=900):
    be = backend()
    if be == "ollama":
        model = model or os.getenv("HARNESS_LLM_MODEL", "qwen3:8b")
        body = {"model": model, "prompt": prompt, "stream": False, "think": False, "format": "json",
                "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": max_tokens}}
        url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req, timeout=timeout))["response"]
    elif be == "openai":
        model = model or os.environ["HARNESS_LLM_MODEL"]
        body = {"model": model, "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
                "temperature": 0, "seed": 0, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        url = os.environ["HARNESS_LLM_URL"].rstrip("/") + "/v1/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"알 수 없는 HARNESS_LLM={be}")
    from harness.extract_llm import parse_salvage
    return parse_salvage(_strip(text))


def _strip(text):
    """사고 블록·코드펜스 제거 — JSON 본문만 남긴다."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()
