#!/usr/bin/env bash
# Build the Assignment-1 submission zip from the tag a1-final (run AFTER tagging, from the repo root, venv active):
#     bash scripts/package_submission.sh [path/to/screen_recording.mp4]
# Produces ../CS203_A1_Project1_23110177_23110297_23110354.zip with the four folders the brief requires:
#   code/             git clone of this repository at tag a1-final, including .git (remote set to GitHub)
#   data/             the DVC-versioned data at a1-final (dvc checkout of the tagged dvc.lock)
#   supporting_files/ supporting_files/ at a1-final + freeze_record.json (+ the screen recording if given)
#   report/           report.pdf
# The working repository is not modified.
set -euo pipefail

NAME="CS203_A1_Project1_23110177_23110297_23110354"
GITHUB="https://github.com/VedantSharan/AgentEvalOps.git"
REPO="$(pwd)"
OUT="$(cd .. && pwd)/${NAME}"
RECORDING="${1:-}"

git rev-parse -q --verify "refs/tags/a1-final" >/dev/null || { echo "Tag a1-final does not exist. Follow HANDOFF_A2.md section 6 first." >&2; exit 1; }
TAG_COMMIT="$(git rev-parse 'a1-final^{commit}')"
[ "$(git rev-parse HEAD)" = "$TAG_COMMIT" ] || { echo "HEAD is not at a1-final. Run: git checkout a1-final" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "Working tree not clean." >&2; exit 1; }
dvc status -q || { echo "dvc status is not clean: run dvc repro / dvc checkout first." >&2; exit 1; }
(cd data/splits/gold_test && sha256sum -c --quiet CHECKSUMS.sha256)

rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT"
git clone -q --branch a1-final "$REPO" "$OUT/code"
git -C "$OUT/code" remote set-url origin "$GITHUB"
cp -r data "$OUT/data"
cp -r supporting_files "$OUT/supporting_files"
mkdir -p "$OUT/report" && cp report/report.pdf "$OUT/report/report.pdf"
if [ -n "$RECORDING" ]; then
  mkdir -p "$OUT/supporting_files/recording" && cp "$RECORDING" "$OUT/supporting_files/recording/"
else
  echo "WARNING: no screen recording given; the brief requires a 2-minute recording in supporting_files/." >&2
fi
python - "$OUT/supporting_files/freeze_record.json" "$TAG_COMMIT" <<'EOF'
import hashlib, json, sys, datetime, yaml, pathlib
out, commit = sys.argv[1], sys.argv[2]
lock = yaml.safe_load(open("dvc.lock"))
outs = {o["path"]: o["md5"] for st in lock["stages"].values() for o in st.get("outs", []) if o["path"].startswith("data/")}
gold = {n: hashlib.sha256(pathlib.Path("data/splits/gold_test", n).read_bytes()).hexdigest() for n in ("gold_tasks.jsonl", "gold_labels.csv")}
json.dump({"tag": "a1-final", "commit": commit, "packaged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "dvc_data_outputs_md5": outs, "gold_test_sha256": gold}, open(out, "w"), indent=2)
EOF
(cd "$(dirname "$OUT")" && zip -qr "$NAME.zip" "$NAME")
echo "Wrote $(dirname "$OUT")/$NAME.zip  (tag a1-final = $TAG_COMMIT)"
echo "Verify: unzip it into a NEW folder, cd ${NAME}/code and run the 5 commands in README.md (option A)."
