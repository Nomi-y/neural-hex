#!/usr/bin/env bash
# Training-run summary — parse the structured log into a report.
#
# Usage:
#   ./training_summary.sh train.log                        # a log file
#   ./training_summary.sh -c hextrain                      # podman logs
#   ./training_summary.sh train.log -o report.md           # custom output
#   ./training_summary.sh train.log -w                     # watch: refresh every REFRESH s
#
# Infers [infer] phase (self-play vs arena) from the enclosing [selfplay] /
# [arena] markers so the two aren't mashed into one misleading mean.
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
    *)  SRC_FILE="$1"; shift ;;
  esac
done

read_logs() {
  if [ -n "$SRC_FILE" ]; then
    cat -- "$SRC_FILE"
  else
    podman logs "$CONTAINER"
  fi
}

# ── Tag every [infer] line with its phase (sp / arena / other) ────────
# Output:  phase_tag gen leaves/s avg_batch peak_batch
tagged_infer() {
  awk '
    /\[selfplay\] fanning/  { sp=1; ar=0 }
    /\[selfplay\] done in/  { sp=0 }
    /\[arena\] [0-9]+ games @.*on [0-9]+ actor/ { ar=1; sp=0 }
    /\[arena\] done in/     { ar=0 }

    /\[infer\] / && /leaves\/s/ {
      # Extract gen
      match($0, /\[gen ([0-9]+)\]/, a); gen = a[1]
      # Extract values
      lps=""; avg=""; peak=""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+$/ && $(i+1) == "leaves/s,") { lps = $i }
        if ($i == "avg" && $(i+1) == "leaves/batch,") { avg = $(i-1) }
        if ($i == "peak" && $(i+1) == "batch") { v = $(i+2); gsub(/,/, "", v); peak = v }
      }
      if (lps != "") {
        tag = sp ? "sp" : (ar ? "arena" : "other")
        printf "%s\t%s\t%s\t%s\t%s\n", tag, gen, lps, avg, peak
      }
    }

    /\[summary\] DONE / {
      delete val; gen=""; res=""
      for (i = 1; i <= NF; i++) {
        if ($i == "[gen") { g = $(i+1); gsub(/]/, "", g); gen = g }
        else if ($i == "PROMOTED" || $i == "kept") { res = $i }
        else if (index($i, "=") > 0) {
          n = index($i, "="); val[substr($i,1,n-1)] = substr($i,n+1)
        }
      }
      if (gen != "") printf "done\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        gen, val["ploss"], val["vloss"], val["arena"], res,
        val["gen_time"], val["elapsed"], val["samples"], val["buffer"]
    }

    /\[hw\] / {
      cpu=""; ram_pct=""; ram_gb=""; gpu=""; vram_used=""; vram_total=""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^cpu=/)  { cpu  = substr($i, 5); gsub(/%/, "", cpu) }
        if ($i ~ /^ram=/)  { v = substr($i, 5); gsub(/%/, "", v); ram_pct = v }
        if ($i ~ /^ram=.*\(/) {
          v = $i; gsub(/.*\(/, "", v); gsub(/G\).*/, "", v); ram_gb = v
        }
        if ($i ~ /^gpu=/)  { v = substr($i, 5); gsub(/%/, "", v); gpu = v }
        if ($i ~ /^vram=/) { split(substr($i,6), a, "/"); vram_used = a[1]; gsub(/MiB.*/, "", a[2]); vram_total = a[2] }
      }
      printf "hw\t%s\t%s\t%s\t%s\t%s\t%s\n", cpu, ram_pct, ram_gb, gpu, vram_used, vram_total
    }

    # Phase timestamps for timeline
    /\[selfplay\] fanning/  { match($0, /\[gen ([0-9]+)\]/, a)
      printf "phase\tsp_start\t%s\t%s\n", a[1], $1 }
    /\[selfplay\] done in/  {
      printf "phase\tsp_done\t%s\t%s\n", (sp_gen?sp_gen:"?"), $1 }
    /\[gen [0-9]+\] \[arena\] [0-9]+ games @/ && !/fanning/ {
      match($0, /\[gen ([0-9]+)\]/, a)
      printf "phase\tarena_start\t%s\t%s\n", a[1], $1 }
    /\[arena\] done in/     {
      printf "phase\tarena_done\t%s\t%s\n", (ar_gen?ar_gen:"?"), $1 }

    # Training progress
    /\[train\] step [0-9]+/ {
      match($0, /step ([0-9]+)/, a); step = a[1]
      match($0, /\(([0-9]+)\/s\)/, a); rate = a[1]
      printf "train_step\t%s\t%s\t%s\n", (t_gen?t_gen:"?"), step, rate
    }
    /\[train\] training.*steps/     { match($0, /\[gen ([0-9]+)\]/, a); t_gen=a[1] }
    /\[selfplay\] fanning/           { match($0, /\[gen ([0-9]+)\]/, a); sp_gen=a[1] }
    /\[arena\] [0-9]+ games @/ && !/fanning/ { match($0, /\[gen ([0-9]+)\]/, a); ar_gen=a[1] }

    # Wall-clock elapsed for timeline
    /\[infer\] / && /leaves\/s/ {
      match($0, /\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9] \+[0-9]+:[0-9][0-9]:[0-9][0-9]\]/)
      ts = substr($0, RSTART+1, RLENGTH-2)
      printf "timeline\t%s\t%s\t%s\n", ts, lps, (sp ? "sp" : (ar ? "arena" : "other"))
    }
  '
}

generate() {
  local raw src_desc tagged
  src_desc=$([ -n "$SRC_FILE" ] && echo "file: $SRC_FILE" || echo "podman container: $CONTAINER")
  raw="$(read_logs 2>&1)" || { echo "Could not read logs ($src_desc)." >&2; return 1; }
  tagged="$(printf '%s\n' "$raw" | tagged_infer)"

  {
    echo "# Training summary"
    echo
    echo "_Generated $(date '+%Y-%m-%d %H:%M:%S') from ${src_desc}._"
    echo

    # ── Hardware utilisation ──────────────────────────────────────────
    echo "## Hardware utilisation"
    echo
    printf '%s\n' "$tagged" | awk -F'\t' '$1=="hw" {
      n++; cs+=$2; rs+=$3; gs+=$5; vs+=$6
      if(n==1||$5<gmin) gmin=$5
      if($5>gmax) gmax=$5
      lrg=$4; lvt=$7
    } END {
      if(n>0) {
        print "| Metric | Min | Mean | Max |"
        print "|--------|----:|-----:|----:|"
        printf "| GPU    | %d%% | %d%% | %d%% |\n", gmin, gs/n, gmax
        printf "| CPU    |  —  | %d%% |  —  |\n", cs/n
        printf "| RAM    |  —  | %d%% (%.0fG) |  —  |\n", rs/n, lrg
        printf "| VRAM   |  —  | %d MiB / %d MiB |  —  |\n", vs/n, lvt
        printf "\n_%d samples at ~60s intervals._\n", n
      }
    }'
    echo

    # ── Inference throughput (split by phase) ─────────────────────────
    echo "## Inference throughput"
    echo

    for phase in sp arena; do
      label=$([ "$phase" = "sp" ] && echo "Self-play" || echo "Arena")
      printf '%s\n' "$tagged" | awk -F'\t' -v ph="$phase" '
        $1 == ph {
          n++; ls+=$3; bs+=$4
          if(n==1||$3<lmin) lmin=$3
          if($3>lmax) lmax=$3
          if($5>pmax) pmax=$5
        }
        END {
          if (n > 0) {
            printf "### %s\n\n", "'"$label"'"
            print "| Metric | Min | Mean | Max |"
            print "|--------|----:|-----:|----:|"
            printf "| leaves/s     | %d | %d | %d |\n", lmin, ls/n, lmax
            printf "| avg batch    |  —  | %d |  —  |\n", bs/n
            printf "| peak batch   |  —  |  —  | %d |\n", pmax
            printf "\n_%d heartbeat lines at ~30s intervals._\n", n
            print ""
          }
        }'
    done

    # ── Per-generation ────────────────────────────────────────────────
    local done_count
    done_count=$(printf '%s\n' "$tagged" | awk -F'\t' '$1=="done"' | grep -c . || echo 0)

    if [ "$done_count" -eq 0 ]; then
      echo "No completed generations yet."
      echo
    else
      echo "## Per-generation"
      echo
      echo "| Gen | ploss | vloss | Arena | Result | Gen time | Samples | Buffer | sp leaves/s | arena leaves/s |"
      echo "|----:|------:|------:|------:|:-------|---------:|--------:|-------:|------------:|--------------:|"

      # Collect all done lines indexed by gen
      printf '%s\n' "$tagged" | awk -F'\t' '$1=="done" {
        g=$2; pl[g]=$3; vl[g]=$4; ap[g]=$5; re[g]=$6; gt[g]=$7; el[g]=$8; sa[g]=$9; bu[g]=$10
        gens[++ngen]=g
      }
      $1=="sp" {
        sp_n[$2]++; sp_s[$2]+=$3
      }
      $1=="arena" {
        ar_n[$2]++; ar_s[$2]+=$3
      }
      END {
        for (i=1; i<=ngen; i++) {
          g = gens[i]
          mark = (re[g] == "PROMOTED") ? "**PROMOTED**" : re[g]
          sp_lps = (sp_n[g] > 0) ? sprintf("%d", sp_s[g]/sp_n[g]) : "—"
          ar_lps = (ar_n[g] > 0) ? sprintf("%d", ar_s[g]/ar_n[g]) : "—"
          printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
            g, pl[g], vl[g], ap[g], mark, gt[g], sa[g], bu[g], sp_lps, ar_lps
        }
      }'
      echo
    fi

    # ── Throughput timeline ───────────────────────────────────────────
    echo "## Throughput timeline (leaves/s over wall time)"
    echo
    echo "| Elapsed | leaves/s | Phase |"
    echo "|---------|---------:|:-----:|"
    printf '%s\n' "$tagged" | awk -F'\t' '$1=="timeline" {
      split($2, a, " ")
      offset = a[2]; gsub(/\+/, "", offset)
      split(offset, o, ":")
      mins = o[1]*60 + o[2] + o[3]/60
      if (!seen || int(mins/2) > int(last_m/2)) {
        seen = 1
        ph = ($4=="arena"?"arena":"self-play")
        printf "| %d min | %s | %s |\n", int(mins), $3, ph
      }
      last_m = mins
    }'
    echo

    # ── Quick stats ───────────────────────────────────────────────────
    echo "## Quick stats"
    echo
    printf '%s\n' "$tagged" | awk -F'\t' '
      $1=="done" {
        gens++
        if (gens == 1) { fp=$3; fv=$4 }
        if ($6=="PROMOTED") promos++
        lp=$3; lv=$4; la=$5; lg=$2; lel=$8; lsa=$9; lbu=$10
      }
      $1=="sp" { spn++; sps+=$3 }
      END {
        printf "- Generations completed: **%d**\n", gens
        printf "- Promotions: **%d**\n", promos+0
        if (spn > 0) printf "- Self-play leaves/s (mean): **%d**\n", sps/spn
        if (gens > 0) {
          printf "- ploss: %s → %s\n", fp, lp
          printf "- vloss: %s → %s  _(falling = value head is learning)_\n", fv, lv
          printf "- Latest arena win-rate: **%s** (gen %s)\n", la, lg
          printf "- Total elapsed: **%s**\n", lel
          printf "- Buffer size: **%s** samples\n", lbu
        }
      }'
    echo

    # ── Raw DONE lines ────────────────────────────────────────────────
    echo "## Raw DONE lines"
    echo
    echo '```'
    printf '%s\n' "$raw" | grep -F '[summary] DONE ' || echo "  (none yet)"
    echo '```'

  } > "$OUT"

  local d=$(printf '%s\n' "$tagged" | awk -F'\t' '$1=="done"' | grep -c . || echo 0)
  local h=$(printf '%s\n' "$tagged" | awk -F'\t' '$1=="hw"'   | grep -c . || echo 0)
  local i=$(printf '%s\n' "$tagged" | awk -F'\t' '$1=="sp"||$1=="arena"||$1=="other"' | grep -c . || echo 0)
  echo "Wrote $d generation(s), $h hw sample(s), $i infer heartbeat(s) to $OUT"
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
