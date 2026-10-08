"""장시간 측정 감독 — OS 무관(supervise.sh 의 Python 판, Windows 에서도 동작).

비정상 종료 시 이어하기로 재시작(최대 N회), 재시작마다 로그에 남긴다. 측정 스크립트는 결과 파일에서
이어하기를 하므로 재시작해도 이미 잰 칸은 다시 재지 않는다.

  python -m harness.supervise --max 10 --log harness/results/x.log -- python -m harness.hybrid ...
"""
import argparse
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    for n in range(a.max + 1):
        with open(a.log, "a") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        with open(a.log, "a") as f:
            if rc == 0:
                f.write(f"[supervise] {time.strftime('%F %T')} 정상 종료\n")
                return 0
            if n == a.max:
                break
            f.write(f"[supervise] {time.strftime('%F %T')} 비정상 종료(rc={rc}) — 재시작 {n + 1}/{a.max}\n")
        time.sleep(30)
    with open(a.log, "a") as f:
        f.write("[supervise] 재시작 한도 도달 — 중단\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
