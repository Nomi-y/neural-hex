#!/usr/bin/env bash
# Extract the per-generation "DONE" lines from a training run and render a summary document for easy
# analysis (a parsed table of ploss/vloss/arena over generations, plus quick stats and the raw lines).
#
# Usage:
#   ./training_summary.sh                       # podman logs hextrain  -> training_summary.md
#   ./training_summary.sh -c mycontainer        # a different container
#   ./training_summary.sh -f train.log          # read a log file instead (nohup/tee users)
#   ./training_summary.sh -o report.md          # choose the output file
#   ./training_summary.sh -w                    # watch: regenerate every REFRESH seconds
#
# The DONE line written by train.py looks like:
#   [23:52:01 +0:45:53] [gen 4] DONE samples=34385 buffer=120000 ploss=4.866 vloss=0.197 \
#       arena=50% kept gen_time=2754s elapsed=0:45:53 remaining=23:14:06 (~30 more gens)
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER="hextrain"
OUT="training_summary.md"
SRC_FILE=""
WATCH=0
REFRESH="${REFRESH:-30}"

while [ $# -gt 0 ]; do
  case "$1" in
    -c|--container) CONTAINER="$2"; shift 2 ;;
    -f|--file)      SRC_FILE="$2";  shift 2 ;;
    -o|--out)       OUT="$2";       shift 2 ;;
    -w|--watch)     WATCH=1;        shift ;;
    -h|--help)      grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $1 (try -h)" >&2; exit 2 ;;
  esac
done

read_logs() {
  if [ -n "$SRC_FILE" ]; then
    cat -- "$SRC_FILE"
  else
    podman logs "$CONTAINER"
  fi
}

# TSV: gen \t ploss \t vloss \t arena \t result \t gen_time \t elapsed
parse_done() {
  awk '
    /\] DONE / {
      delete val; gen=""; res=""
      for (i = 1; i <= NF; i++) {
        if ($i == "[gen") { g = $(i+1); gsub(/]/, "", g); gen = g }
        else if ($i == "PROMOTED" || $i == "kept") { res = $i }
        else if (index($i, "=") > 0) {
          n = index($i, "="); val[substr($i,1,n-1)] = substr($i,n+1)
        }
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        gen, val["ploss"], val["vloss"], val["arena"], res, val["gen_time"], val["elapsed"]
    }'
}

generate() {
  local raw done_lines tsv src_desc
  src_desc=$([ -n "$SRC_FILE" ] && echo "file: $SRC_FILE" || echo "podman container: $CONTAINER")
  raw="$(read_logs 2>&1)" || { echo "Could not read logs ($src_desc)." >&2; return 1; }
  done_lines="$(printf '%s\n' "$raw" | grep -F '] DONE ' || true)"
  tsv="$(printf '%s\n' "$done_lines" | parse_done)"

  {
    echo "# Training summary"
    echo
    echo "_Generated $(date '+%Y-%m-%d %H:%M:%S') from ${src_desc}._"
    echo

    if [ -z "$(printf '%s' "$tsv" | tr -d '[:space:]')" ]; then
      echo "No completed generations yet (no \`DONE\` lines found — gen 1 may still be in progress)."
      return 0
    fi

    echo "## Per-generation"
    echo
    printf '%s\n' "$tsv" | awk -F'\t' '
      BEGIN {
        print "| Gen | ploss | vloss | Arena | Result | Gen time | Elapsed |"
        print "|----:|------:|------:|------:|:-------|---------:|:--------|"
      }
      { mark = ($5 == "PROMOTED") ? "**PROMOTED**" : $5
        printf "| %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, mark, $6, $7 }'
    echo

    echo "## Quick stats"
    echo
    printf '%s\n' "$tsv" | awk -F'\t' '
      { gens++; if ($5 == "PROMOTED") promos++
        last_p=$2; last_v=$3; last_a=$4; last_g=$1
        if (NR == 1) { first_p=$2; first_v=$3 } }
      END {
        printf "- Generations completed: **%d**\n", gens
        printf "- Promotions: **%d**\n", promos+0
        printf "- ploss: %s → %s\n", first_p, last_p
        printf "- vloss: %s → %s  _(falling = value head is learning)_\n", first_v, last_v
        printf "- Latest arena win-rate: **%s** (at gen %s)\n", last_a, last_g
      }'
    echo

    echo "## Raw DONE lines"
    echo
    echo '```'
    printf '%s\n' "$done_lines"
    echo '```'
  } > "$OUT"

  echo "Wrote $(printf '%s\n' "$tsv" | grep -c . ) generation(s) to $OUT"
}

if [ "$WATCH" -eq 1 ]; then
  echo "Watching; refreshing $OUT every ${REFRESH}s. Ctrl-C to stop."
  while true; do
    generate || true
    sleep "$REFRESH"
  done
else
  generate
fi
