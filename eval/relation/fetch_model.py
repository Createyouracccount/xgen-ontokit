"""MODEL_LOCK.json 기준 관계 인코더 가중치 다운로드·검증.

가중치는 git 미커밋(261MB~) — GitHub Release 자산으로 배포하고, 이 스크립트가
잠금파일의 sha256 을 검증해 받는다. "커밋 = 모델 버전 배포"의 실행부:
  잠금파일 갱신 커밋 → 사용처는 fetch_model.py 1회 → env 로 로드.

사용: /opt/miniconda3/bin/python fetch_model.py [--dest DIR] [--profile 이름]
  --profile 미지정 = default(re-ko-hard-v13c, 속도).
  --profile quality-large = re-ko-large-v1(정밀도, 빌드 ~2.2배 — MODEL_LOCK profiles 참고).
완료 후: export ONTOKIT_RELATION_ENCODER_MODEL=<압축해제 디렉토리>
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile

SP = pathlib.Path(__file__).parent
LOCK = json.load(open(SP / "MODEL_LOCK.json"))

profile = sys.argv[sys.argv.index("--profile") + 1] if "--profile" in sys.argv else "default"
profiles = LOCK.get("profiles", {})
sel = profiles.get(profile, "relation_encoder_ko" if profile == "default" else None)
if sel is None or (isinstance(sel, str) and sel not in LOCK):
    sys.exit(f"프로파일 없음: {profile} (가능: default, "
             + ", ".join(k for k in profiles if not k.startswith('_') and k != 'default') + ")")
lock = LOCK[sel] if isinstance(sel, str) else sel

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
out = dest / lock.get("extract_dir", "model_re_aug")
print(f"OK → {out}  (env {lock['env']} 로 지정)")
