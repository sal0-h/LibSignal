#!/usr/bin/env bash
# Resubmit unfinished Ingolstadt axis ablations (after i21_axes job 8727).
#
# Job 8727 finished hetero + slow_start (10/25). Failed on crossing_proxy;
# obs and noise never ran.
#
# Usage (gpujobs):
#   export MCS_LABEL=15288
#   ./extras/resubmit_ingolstadt_axes_remainder.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export JOB_GROUPS=axes
export AXES_FILTER="${AXES_FILTER:-crossing_proxy,obs,noise}"
exec ./extras/submit_ingolstadt_1x21_chained.sh
