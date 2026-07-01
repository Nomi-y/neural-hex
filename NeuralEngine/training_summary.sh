#!/usr/bin/env bash
# Training-run summary — parse the structured log into a report.
#
# Usage:
#   ./training_summary.sh train_20260630_231305.log          # a log file
#   ./training_summary.sh -c hextrain                        # podman logs
#   ./training_summary.sh train.log -o report.md             # custom output
#   ./training_summary.sh train.log -w                       # watch: refresh every REFRESH s
#
# Reads [gen N[summary] DONE … lines (generation table), [hw] … lines (hardware
# utilisation timeline), and [infer] … lines (inference-server throughput).
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
    -*) echo "Unknown flag: $1 (try -h)" >&2; exit 2 ;;
    *)  SRC_FILE="$1"; shift ;;   # positional arg = log file
  esac
done

read_logs() {
  if [ -n "$SRC_FILE" ]; then
    cat -- "$SRC_FILE"
  else
    podman logs "$CONTAINER"
  fi
}

# ── DONE lines → TSV: gen \t ploss \t vloss \t arena \t result \t gen_time \t elapsed \t samples
parse_done() {
  awk '
    /\[summary] DONE / {
      delete val; gen=""; res=""
      for (i = 1; i <= NF; i++) {
        if ($i == "[gen") { g = $(i+1); gsub(/]/, "", g); gen = g }
        else if ($i == "PROMOTED" || $i == "kept") { res = $i }
        else if (index($i, "=") > 0) {
          n = index($i, "="); val[substr($i,1,n-1)] = substr($i,n+1)
        }
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        gen, val["ploss"], val["vloss"], val["arena"], res,
        val["gen_time"], val["elapsed"], val["samples"]
    }'
}

# ── [hw] lines → TSV: cpu_pct \t ram_pct \t ram_gb \t gpu_pct \t vram_used \t vram_total
parse_hw() {
  awk '
    /\[hw\]/ {
      cpu=""; ram_pct=""; ram_gb=""; gpu=""; vram_used=""; vram_total=""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^cpu=/)  { cpu  = substr($i, 5) }
        if ($i ~ /^ram=/)  { v = substr($i, 5); gsub(/%/, "", v); ram_pct = v }
        if ($i ~ /^ram=.*\(/) {
          v = $i; gsub(/.*\(/, "", v); gsub(/G\).*/, "", v); ram_gb = v
        }
        if ($i ~ /^gpu=/)  { v = substr($i, 5); gsub(/%/, "", v); gpu = v }
        if ($i ~ /^vram=/) { split(substr($i,6), a, "/"); vram_used = a[1]; gsub(/MiB.*/, "", a[2]); vram_total = a[2] }
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\n", cpu, ram_pct, ram_gb, gpu, vram_used, vram_total
    }'
}

# ── [infer] lines → TSV: leaves_per_s \t avg_batch \t peak_batch
parse_infer() {
  awk '
    /\[infer\]/ && /leaves\/s/ {
      lps=""; avg=""; peak=""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+$/ && $(i+1) == "leaves/s,") { lps = $i }
        if ($i == "avg" && $(i+1) == "leaves/batch,") { avg = $(i-1) }
        if ($i == "peak" && $(i+1) == "batch") { v = $(i+2); gsub(/,/, "", v); peak = v }
      }
      if (lps != "") printf "%s\t%s\t%s\n", lps, avg, peak
    }'
}

generate() {
  local raw src_desc
  src_desc=$([ -n "$SRC_FILE" ] && echo "file: $SRC_FILE" || echo "podman container: $CONTAINER")
  raw="$(read_logs 2>&1)" || { echo "Could not read logs ($src_desc)." >&2; return 1; }

  local done_lines hw_lines infer_lines done_tsv hw_tsv infer_tsv
  done_lines="$(printf '%s\n' "$raw"  | grep -F '[summary] DONE '  || true)"
  hw_lines="$(printf   '%s\n' "$raw"  | grep -F '[hw] '     || true)"
  infer_lines="$(printf '%s\n' "$raw" | grep -F '[infer] '  | grep 'leaves/s' || true)"
  done_tsv="$(printf  '%s\n' "$done_lines"  | parse_done)"
  hw_tsv="$(printf    '%s\n' "$hw_lines"    | parse_hw)"
  infer_tsv="$(printf '%s\n' "$infer_lines" | parse_infer)"

  {
    echo "# Training summary"
    echo
    echo "_Generated $(date '+%Y-%m-%d %H:%M:%S') from ${src_desc}._"
    echo

    # ── Hardware utilisation ──────────────────────────────────────────
    if [ -n "$(printf '%s' "$hw_tsv" | tr -d '[:space:]')" ]; then
      echo "## Hardware utilisation"
      echo
      printf '%s\n' "$hw_tsv" | awk -F'\t' '
        BEGIN {
          gpu_min=999; gpu_max=0; gpu_sum=0
          cpu_sum=0; ram_sum=0; vram_sum=0; n=0
        }
        {
          n++
          cpu_sum  += $1
          ram_sum  += $2
          gpu_sum  += $4
          vram_sum += $5
          if ($4 < gpu_min) gpu_min = $4
          if ($4 > gpu_max) gpu_max = $4
          last_ram_gb = $3; last_vram_total = $6
        }
        END {
          if (n > 0) {
            printf "| Metric | Min | Mean | Max |\n"
            printf "|--------|----:|-----:|----:|\n"
            printf "| GPU    | %d%% | %d%% | %d%% |\n", gpu_min, gpu_sum/n, gpu_max
            printf "| CPU    |  —  | %d%% |  —  |\n", cpu_sum/n
            printf "| RAM    |  —  | %d%% (%.0fG) |  —  |\n", ram_sum/n, last_ram_gb
            printf "| VRAM   |  —  | %d MiB / %d MiB |  —  |\n", vram_sum/n, last_vram_total
            printf "\n_%d samples at ~60s intervals._\n", n
          }
        }'
      echo
    fi

    # ── Inference throughput ──────────────────────────────────────────
    if [ -n "$(printf '%s' "$infer_tsv" | tr -d '[:space:]')" ]; then
      echo "## Inference server throughput"
      echo
      printf '%s\n' "$infer_tsv" | awk -F'\t' '
        BEGIN {
          lps_min=999999; lps_max=0; lps_sum=0
          batch_sum=0; peak_max=0; n=0
        }
        {
          n++
          lps_sum += $1
          batch_sum += $2
          if ($1 < lps_min) lps_min = $1
          if ($1 > lps_max) lps_max = $1
          if ($3 > peak_max) peak_max = $3
        }
        END {
          if (n > 0) {
            printf "| Metric | Min | Mean | Max |\n"
            printf "|--------|----:|-----:|----:|\n"
            printf "| leaves/s     | %d | %d | %d |\n", lps_min, lps_sum/n, lps_max
            printf "| avg batch    |  —  | %d |  —  |\n", batch_sum/n
            printf "| peak batch   |  —  |  —  | %d |\n", peak_max
            printf "\n_%d heartbeat lines at ~30s intervals._\n", n
          }
        }'
      echo
    fi

    # ── Per-generation ────────────────────────────────────────────────
    if [ -z "$(printf '%s' "$done_tsv" | tr -d '[:space:]')" ]; then
      echo "No completed generations yet (no \`DONE\` lines found — gen 1 may still be in progress)."
      return 0
    fi

    echo "## Per-generation"
    echo
    printf '%s\n' "$done_tsv" | awk -F'\t' '
      BEGIN {
        print "| Gen | ploss | vloss | Arena | Result | Gen time | Samples |"
        print "|----:|------:|------:|------:|:-------|---------:|--------:|"
      }
      { mark = ($5 == "PROMOTED") ? "**PROMOTED**" : $5
        printf "| %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, mark, $6, $8 }'
    echo

    echo "## Quick stats"
    echo
    printf '%s\n' "$done_tsv" | awk -F'\t' '
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

  echo "Wrote $(printf   '%s\n' "$done_tsv"  | grep -c . || echo 0) generation(s),"
  echo "      $(printf   '%s\n' "$hw_tsv"    | grep -c . || echo 0) hw sample(s),"
  echo "      $(printf   '%s\n' "$infer_tsv" | grep -c . || echo 0) infer heartbeat(s)"
  echo "to $OUT"
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
