#!/usr/bin/env bash
# Safe git pull on gpujobs when untracked experiment logs block checkout.
#
# Those BRF/DTL files are already committed on feat/early-stop-episodes.
# Untracked copies on the server block pull — this backs them up, then pulls.
#
# Usage (gpujobs):
#   cd ~/LibSignalFork
#   bash extras/gpujobs_pull_safe.sh
#   bash extras/smoke_crossing_proxy_1x21.sh
#   export MCS_LABEL=15288 && ./extras/resubmit_ingolstadt_axes_remainder.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-feat/early-stop-episodes}"
BACKUP="${BACKUP:-$HOME/LibSignalFork_untracked_backup_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BACKUP"

backup_if_untracked() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    echo "tracked (will merge): $f"
    return 0
  fi
  echo "backup untracked: $f -> $BACKUP/$f"
  mkdir -p "$BACKUP/$(dirname "$f")"
  cp -a "$f" "$BACKUP/$f"
  rm -f "$f"
}

# Known blockers on feat/early-stop-episodes (also scan for any other untracked logs)
backup_if_untracked "data/output_data/tsc/sumo_presslight_odh_l2/sumo4x4/odh_l2_es/logger/2026_08_05-00_33_53_BRF.log"
backup_if_untracked "data/output_data/tsc/sumo_presslight_odh_l2/sumo4x4/odh_l2_es/logger/2026_08_05-00_33_53_DTL.log"
backup_if_untracked "extras/resubmit_ingolstadt_axes_remainder.sh"

echo "Fetching origin/${BRANCH}..."
git fetch origin "$BRANCH"

# Any untracked file that would be overwritten by merge
while IFS= read -r line; do
  f="${line#Would remove }"
  backup_if_untracked "$f"
done < <(git merge-tree "$(git merge-base HEAD "origin/${BRANCH}")" HEAD "origin/${BRANCH}" 2>/dev/null | grep '^Would remove' || true)

git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/${BRANCH}"
git pull origin "$BRANCH"

chmod +x extras/resubmit_ingolstadt_axes_remainder.sh extras/smoke_crossing_proxy_1x21.sh 2>/dev/null || true

echo ""
echo "Pull OK on branch ${BRANCH}."
echo "Untracked copies backed up to: ${BACKUP}"
echo "Committed logs are in the repo under data/output_data/tsc/..."
echo ""
echo "Compare backup vs repo (optional):"
echo "  diff -q ${BACKUP}/data/output_data/tsc/sumo_presslight_odh_l2/sumo4x4/odh_l2_es/logger/2026_08_05-00_33_53_BRF.log \\"
echo "    data/output_data/tsc/sumo_presslight_odh_l2/sumo4x4/odh_l2_es/logger/2026_08_05-00_33_53_BRF.log || true"
