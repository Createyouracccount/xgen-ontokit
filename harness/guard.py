"""메모리 가드 — OS 무관(Linux·컨테이너·macOS·Windows). memguard.sh(macOS 전용)를 대체한다.

여유 메모리 지표는 OS 마다 의미가 달라 **그 OS 에서 믿을 만한 것**을 고른다(판단 로직은 하나):
  Linux 컨테이너  : cgroup 한도 기준(memory.max·memory.current / v1 limit·usage) — 호스트 여유가 아니라 내 한도
  Linux 호스트    : /proc/meminfo MemAvailable / MemTotal
  macOS           : `memory_pressure` 의 free % (압축 메모리를 반영한 커널 지표; psutil available 은 과소평가)
  Windows         : GlobalMemoryStatusEx ullAvailPhys / ullTotalPhys (psutil 이 있으면 psutil)

동작(퍼센트 = 여유 비율):
  < pause(기본 20)  : 측정 클라이언트 일시정지(psutil suspend — OS 무관)
  ≥ resume(기본 30) : 재개
  < kill(기본 10)   : LLM 서버 종료(캐시·KV 회수) — 감독 스크립트가 측정을 이어서 재시작
모든 조치는 로그에 남는다.

  python -m harness.guard --log harness/results/guard.log \
      [--target harness.hybrid] [--kill mlx_lm.server] [--pause 20 --resume 30 --kill-below 10]
"""
import argparse
import os
import platform
import re
import subprocess
import time


# ── 지표 읽기(파서는 순수 함수 — 단위테스트 대상) ──

def parse_meminfo(text):
    kv = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)", line)
        if m:
            kv[m.group(1)] = int(m.group(2))
    return 100.0 * kv["MemAvailable"] / kv["MemTotal"]


def parse_cgroup(limit_text, usage_text, host_total_bytes):
    """cgroup 한도 기준 여유 %. 한도 없음('max' 또는 호스트보다 큼)이면 None."""
    lim = limit_text.strip()
    if lim == "max":
        return None
    limit = int(lim)
    if limit <= 0 or limit >= host_total_bytes:
        return None
    return 100.0 * max(0, limit - int(usage_text.strip())) / limit


def parse_memory_pressure(text):
    m = re.search(r"free percentage:\s*(\d+)%", text)
    return float(m.group(1)) if m else None


def _read(path):
    with open(path) as f:
        return f.read()


def free_percent():
    """(여유 %, 지표 출처)."""
    system = platform.system()
    if system == "Linux":
        host_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        for lim, use in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
            if os.path.exists(lim) and os.path.exists(use):
                v = parse_cgroup(_read(lim), _read(use), host_total)
                if v is not None:
                    return v, "cgroup"
        return parse_meminfo(_read("/proc/meminfo")), "meminfo"
    if system == "Darwin":
        v = parse_memory_pressure(subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout)
        if v is not None:
            return v, "memory_pressure"
    try:
        import psutil
        m = psutil.virtual_memory()
        return 100.0 * m.available / m.total, "psutil"
    except ImportError:
        pass
    if system == "Windows":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS()
        s.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return 100.0 * s.ullAvailPhys / s.ullTotalPhys, "GlobalMemoryStatusEx"
    raise RuntimeError(f"여유 메모리를 읽을 방법이 없다({system}) — 가드 없이 측정하지 않는다")


# ── 판단(순수 함수) ──

def decide(free, paused, pause=20.0, resume=30.0, kill_below=10.0):
    """→ (조치 목록, 새 paused 상태). 조치 ∈ {"kill", "pause", "resume"}."""
    acts = []
    if free < kill_below:
        acts.append("kill")
    if free < pause and not paused:
        acts.append("pause")
        paused = True
    elif free >= resume and paused:
        acts.append("resume")
        paused = False
    return acts, paused


# ── 프로세스 조치(psutil — OS 무관) ──

def _procs(pattern):
    import psutil
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.pid != me and pattern in " ".join(p.info["cmdline"] or []):
                yield p
        except Exception:
            continue


def act(action, target, kill):
    n = 0
    for p in _procs(kill if action == "kill" else target):
        try:
            {"pause": p.suspend, "resume": p.resume, "kill": p.terminate}[action]()
            n += 1
        except Exception:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="harness/results/guard.log")
    ap.add_argument("--target", default="harness.hybrid")
    ap.add_argument("--kill", default="mlx_lm.server")
    ap.add_argument("--pause", type=float, default=float(os.getenv("GUARD_PAUSE", 20)))
    ap.add_argument("--resume", type=float, default=float(os.getenv("GUARD_RESUME", 30)))
    ap.add_argument("--kill-below", type=float, default=float(os.getenv("GUARD_KILL", 10)))
    ap.add_argument("--interval", type=float, default=30)
    a = ap.parse_args()
    paused = False
    free, src = free_percent()
    with open(a.log, "a") as f:
        f.write(f"{time.strftime('%F %T')} 가드 시작 — 지표 {src}, 현재 여유 {free:.0f}%, "
                f"정지<{a.pause} 재개≥{a.resume} 회수<{a.kill_below}\n")
    while True:
        free, src = free_percent()
        acts, paused = decide(free, paused, a.pause, a.resume, a.kill_below)
        for x in acts:
            n = act(x, a.target, a.kill)
            with open(a.log, "a") as f:
                f.write(f"{time.strftime('%F %T')} 여유 {free:.0f}%({src}) — {x} {n}개 프로세스\n")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
