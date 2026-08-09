#!/usr/bin/env bash
# Safe git pull on gpujobs: keeps experiment logs, discards stale local edits, fast-forwards.
#
# OD-hub BRF/DTL logs are already committed on feat/early-stop-episodes.
# Server often has untracked copies + old edits to world_sumo.py — both block pull.
#
# Usage (gpujobs):
#   cd ~/LibSignalFork
#   bash extras/gpujobs_pull_safe.sh
#
# Recover pre-pull tarball if needed:
#   tar xzf ~/LibSignalFork_pre_pull_<timestamp>.tar.gz -C /tmp/recover

set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-feat/early-stop-episodes}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${ARCHIVE:-$HOME/LibSignalFork_pre_pull_${STAMP}.tar.gz}"

echo "=== LibSignal safe pull (branch ${BRANCH}) ==="

# 1) Full backup of experiment outputs + any local script edits
echo "Archiving to ${ARCHIVE} ..."
tar czf "${ARCHIVE}" \
  data/output_data/tsc \
  world/world_sumo.py \
  extras/submit_ingolstadt_1x21_chained.sh \
  2>/dev/null || true
echo "Backup archive: ${ARCHIVE}"

echo "Fetching origin/${BRANCH}..."
git fetch origin "$BRANCH"

# 2) Drop local edits to tracked files (remote branch has the correct versions)
if git diff --quiet world/world_sumo.py 2>/dev/null; then :; else
  echo "Reverting local edits: world/world_sumo.py"
  git checkout -- world/world_sumo.py
fi
if git diff --quiet extras/submit_ingolstadt_1x21_chained.sh 2>/dev/null; then :; else
  echo "Reverting local edits: extras/submit_ingolstadt_1x21_chained.sh"
  git checkout -- extras/submit_ingolstadt_1x21_chained.sh
fi

# 3) Remove untracked duplicates of log files that exist in the branch
removed=0
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  if [[ -f "$f" ]] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    echo "remove untracked duplicate: $f"
    rm -f "$f"
    removed=$((removed + 1))
  fi
done < <(git ls-tree -r --name-only "origin/${BRANCH}" -- data/output_data/tsc 2>/dev/null | grep '\.log$' || true)
echo "Removed ${removed} untracked log duplicate(s)."

# 4) Fast-forward
current="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
if [[ "$current" != "$BRANCH" ]]; then
  git checkout "$BRANCH" 2>/dev/null || git checkout -B "$BRANCH" "origin/${BRANCH}"
fi
git pull --ff-only origin "$BRANCH"

chmod +x extras/resubmit_ingolstadt_axes_remainder.sh \
         extras/smoke_crossing_proxy_1x21.sh \
         extras/gpujobs_pull_safe.sh 2>/dev/null || true

echo ""
echo "Pull OK. Branch ${BRANCH} is up to date."
echo "Experiment logs restored from git (same runs as your untracked copies)."
echo "Full pre-pull backup: ${ARCHIVE}"
echo ""
echo "Next:"
echo "  bash extras/smoke_crossing_proxy_1x21.sh"
echo "  export MCS_LABEL=15288 && ./extras/resubmit_ingolstadt_axes_remainder.sh"
