"""MODEL_LOCK.json 기준 관계 인코더 가중치 다운로드·검증.

가중치는 git 미커밋(261MB) — GitHub Release 자산으로 배포하고, 이 스크립트가
잠금파일의 sha256 을 검증해 받는다. "커밋 = 모델 버전 배포"의 실행부:
  잠금파일 갱신 커밋 → 사용처는 fetch_model.py 1회 → env 로 로드.

사용: /opt/miniconda3/bin/python fetch_model.py [--dest DIR]
완료 후: export ONTOKIT_RELATION_ENCODER_MODEL=<dest>/model_re_aug
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile

SP = pathlib.Path(__file__).parent
lock = json.load(open(SP / "MODEL_LOCK.json"))["relation_encoder_ko"]
dest = pathlib.Path(sys.argv[sys.argv.index("--dest") + 1]) if "--dest" in sys.argv else SP
tag, asset = lock["release_tag"], lock["asset"]
tarball = dest / asset

if not tarball.exists():
    print(f"다운로드: release {tag} / {asset}")
    subprocess.run(["gh", "release", "download", tag, "--pattern", asset,
                    "--dir", str(dest)], check=True, cwd=SP)

h = hashlib.sha256()
with open(tarball, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != lock["asset_sha256"]:
    sys.exit(f"sha256 불일치: {h.hexdigest()} != {lock['asset_sha256']}")
with tarfile.open(tarball) as t:
    t.extractall(dest)
print(f"OK → {dest / 'model_re_aug'}  (env {lock['env']} 로 지정)")
