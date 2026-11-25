#!/usr/bin/env bash
# Archive the committed code/report, with a local video or explicit live exemption.
# bash scripts/make_submission_zip.sh [OUT.zip] --video /path/to/video.mp4
# bash scripts/make_submission_zip.sh [OUT.zip] --live
# bash scripts/make_submission_zip.sh [OUT.zip] --code-only  # incomplete deliverable
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT_DIR/dist/yanfaxiang_50051798_iota5201.zip"
MODE=""
VIDEO=""
if [[ $# -gt 0 && "$1" != --* ]]; then OUT="$1"; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --video) [[ $# -ge 2 ]] || { echo '--video needs a file' >&2; exit 1; }; MODE=video; VIDEO="$2"; shift 2 ;;
        --live) MODE=live; shift ;;
        --code-only) MODE=code-only; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$MODE" ]] || { echo 'Choose --video FILE, --live (if selected), or --code-only.' >&2; exit 1; }
[[ "$MODE" != video || -f "$VIDEO" ]] || { echo "Missing video: $VIDEO" >&2; exit 1; }
cd "$ROOT_DIR"
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit the intended changes first; packaging uses HEAD.' >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
ARCHIVE_TEMP="$(mktemp "$(dirname "$OUT")/iota-archive-XXXXXX.zip")"
trap 'rm -f "$ARCHIVE_TEMP"' EXIT
git archive --format=zip --prefix=yanfaxiang_50051798_iota5201/ --output="$ARCHIVE_TEMP" HEAD
python3 - "$ARCHIVE_TEMP" "$MODE" "$VIDEO" <<'PY'
import json
from pathlib import Path
import sys
import zipfile
archive, mode, video = sys.argv[1:]
prefix = 'yanfaxiang_50051798_iota5201/'
with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as package:
    if mode == 'video':
        package.write(video, prefix+'presentation_video'+Path(video).suffix)
    package.writestr(prefix+'SUBMISSION_CONTENTS.json', json.dumps({
        'mode': mode,
        'contains_video': mode == 'video',
        'video_exempt_selected_for_live_presentation': mode == 'live',
        'complete_deliverable': mode != 'code-only',
    }, indent=2)+'\n')
    required = ['README.md', 'paper/main.pdf', 'configs/grpo_ablation.json',
                'refinement_plugin.py', 'src/refinement_rewards.py', 'scripts/run_refinement.py']
    for path in required:
        if prefix+path not in package.namelist():
            raise RuntimeError('Missing required code/report: '+path)
    if package.testzip():
        raise RuntimeError('ZIP integrity check failed')
PY
mv "$ARCHIVE_TEMP" "$OUT"
echo "Written: $OUT (mode: $MODE)"
