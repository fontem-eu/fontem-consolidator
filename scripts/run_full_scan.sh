#!/usr/bin/env bash
# Ad-hoc full-graph consolidation scan.
#
# 1. Snapshots every :Company gmr_id and :Authority authority_id into
#    .scan_state/{companies,authorities}.todo (only on first run, or when
#    --refresh is passed).
# 2. Iterates line-by-line, calls the consolidator API for each id, and
#    deletes the processed line from the file so the scan is resumable.
# 3. Sleeps between calls so Neo4j doesn't get hammered.
#
# Usage:
#   ./run_full_scan.sh                     # resume or start
#   ./run_full_scan.sh --refresh           # re-snapshot ids (discards .todo files)
#   BATCH_SIZE=20 SLEEP_SECS=2 ./run_full_scan.sh
#
# Env:
#   SLEEP_SECS      default 1   (gap between batches)
#   BATCH_SIZE      default 10  (ids per POST /consolidate/batch)
#   NEO4J_NS        default gmr
#   CONSOLIDATOR_NS default gmr
#   STATE_DIR       default ./.scan_state

set -eu

SLEEP_SECS="${SLEEP_SECS:-1}"
BATCH_SIZE="${BATCH_SIZE:-10}"
SUMMARY_EVERY="${SUMMARY_EVERY:-20}"   # batches between summary lines
EXCLUDE_RULE_PREFIX="${EXCLUDE_RULE_PREFIX:-gds_}"
NEO4J_NS="${NEO4J_NS:-gmr}"
CONSOLIDATOR_NS="${CONSOLIDATOR_NS:-gmr}"
STATE_DIR="${STATE_DIR:-.scan_state}"

# Aggregate counters across the lifetime of this process
TOTAL_PROCESSED=0
TOTAL_MERGED=0
TOTAL_FLAGGED=0
TOTAL_CONFLICTS=0
TOTAL_FAILED=0
START_EPOCH=$(date +%s)

REFRESH=0
[[ "${1:-}" == "--refresh" ]] && REFRESH=1

mkdir -p "$STATE_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

snapshot_ids() {
  local kind="$1"   # Company | Authority
  local id_prop="$2" # gmr_id | authority_id
  local out="$3"
  log "snapshotting $kind ids → $out"
  local neo_pod
  neo_pod=$(kubectl get pods -n "$NEO4J_NS" -l app=neo4j -o jsonpath='{.items[0].metadata.name}')
  kubectl exec -n "$NEO4J_NS" "$neo_pod" -- sh -c "
    NEO4J_PWD=\$(echo \$NEO4J_AUTH | cut -d/ -f2)
    echo 'MATCH (n:$kind) WHERE n.$id_prop IS NOT NULL RETURN n.$id_prop;' \
      | cypher-shell -u neo4j -p \"\$NEO4J_PWD\" -d neo4j --format plain 2>/dev/null
  " | tail -n +2 | tr -d '"' > "$out"
  local count
  count=$(wc -l < "$out" | tr -d ' ')
  log "  $count ids written"
}

ensure_snapshot() {
  local kind="$1" id_prop="$2" file="$3"
  if [[ $REFRESH -eq 1 ]] || [[ ! -f "$file" ]]; then
    snapshot_ids "$kind" "$id_prop" "$file"
  fi
}

process_batch() {
  # $1 = entity_type (Company|Authority)
  # $2 = newline-separated ids (max BATCH_SIZE)
  # stdout: short status string
  local entity_type="$1" ids_json
  ids_json=$(printf '%s\n' "$2" | python3 -c 'import sys,json; print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))')
  kubectl exec -n "$CONSOLIDATOR_NS" deployment/gmr-consolidator -c gmr-consolidator -- \
    python3 -c "
import sys, json, urllib.request, urllib.error
body = json.dumps({'entity_type': '$entity_type', 'ids': $ids_json, 'triggered_by': 'full_scan', 'exclude_rule_prefix': '$EXCLUDE_RULE_PREFIX' or None}).encode()
req = urllib.request.Request('http://localhost:8000/consolidate/batch', data=body, method='POST',
                             headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
        print('ok processed={} merged={} flagged={} conflicts={}'.format(
            d.get('processed',0), d.get('merged',0), d.get('flagged',0), d.get('conflicts',0)))
except urllib.error.HTTPError as e:
    print('http', e.code); sys.exit(1)
except Exception as e:
    print('err', str(e)[:80]); sys.exit(1)
" 2>/dev/null || return 1
}

parse_counters() {
  # Extract processed/merged/flagged/conflicts from a status line.
  # Returns via stdout: "<processed> <merged> <flagged> <conflicts>"
  local status="$1"
  local p=$(echo "$status" | grep -oE 'processed=[0-9]+' | cut -d= -f2)
  local m=$(echo "$status" | grep -oE 'merged=[0-9]+'    | cut -d= -f2)
  local f=$(echo "$status" | grep -oE 'flagged=[0-9]+'   | cut -d= -f2)
  local c=$(echo "$status" | grep -oE 'conflicts=[0-9]+' | cut -d= -f2)
  printf '%d %d %d %d' "${p:-0}" "${m:-0}" "${f:-0}" "${c:-0}"
}

emit_summary() {
  local entity_type="$1" file="$2"
  local now=$(date +%s)
  local elapsed=$((now - START_EPOCH))
  local rate=$(awk -v p="$TOTAL_PROCESSED" -v e="$elapsed" 'BEGIN{ if(e>0) printf "%.2f", p/e; else print "0" }')
  local remaining=$(wc -l < "$file" | tr -d ' ')
  local eta="n/a"
  if [[ "$rate" != "0" && "$remaining" -gt 0 ]]; then
    eta=$(awk -v r="$remaining" -v rt="$rate" 'BEGIN{ s=r/rt; h=int(s/3600); m=int((s-h*3600)/60); printf "%dh%02dm", h, m }')
  fi
  printf '[%s] SUMMARY %s: processed=%d merged=%d flagged=%d conflicts=%d failed=%d rate=%s/s remaining=%d eta=%s\n' \
    "$(date +%H:%M:%S)" "$entity_type" "$TOTAL_PROCESSED" "$TOTAL_MERGED" "$TOTAL_FLAGGED" "$TOTAL_CONFLICTS" "$TOTAL_FAILED" "$rate" "$remaining" "$eta"
}

drain_file() {
  # $1 = entity_type (Company|Authority)
  # $2 = .todo file
  local entity_type="$1" file="$2" total done_n=0 batch_idx=0
  total=$(wc -l < "$file" | tr -d ' ')
  log "draining $file ($total remaining, batch=$BATCH_SIZE sleep=${SLEEP_SECS}s)"
  while [[ -s "$file" ]]; do
    local ids
    ids=$(head -n "$BATCH_SIZE" "$file" | grep -v '^$' || true)
    [[ -z "$ids" ]] && { : > "$file"; break; }
    local n
    n=$(printf '%s\n' "$ids" | wc -l | tr -d ' ')
    local status
    status=$(process_batch "$entity_type" "$ids" || echo "fail")
    done_n=$((done_n + n))
    batch_idx=$((batch_idx + 1))

    if [[ "$status" == fail* || "$status" == err* || "$status" == http* ]]; then
      TOTAL_FAILED=$((TOTAL_FAILED + n))
      printf '[%s] FAIL   %s batch=%d ids=%d status=%s\n' "$(date +%H:%M:%S)" "$entity_type" "$batch_idx" "$n" "$status"
    else
      read -r p m f c < <(parse_counters "$status")
      TOTAL_PROCESSED=$((TOTAL_PROCESSED + p))
      TOTAL_MERGED=$((TOTAL_MERGED + m))
      TOTAL_FLAGGED=$((TOTAL_FLAGGED + f))
      TOTAL_CONFLICTS=$((TOTAL_CONFLICTS + c))
    fi

    # Drop processed lines (1..$BATCH_SIZE)
    sed -i "1,${BATCH_SIZE}d" "$file"

    if (( batch_idx % SUMMARY_EVERY == 0 )); then
      emit_summary "$entity_type" "$file"
    fi
    sleep "$SLEEP_SECS"
  done
  emit_summary "$entity_type" "$file"
  log "$entity_type done ($done_n processed this run)"
}

on_exit() {
  log "interrupted — emitting final summary"
  emit_summary "TOTAL" "$STATE_DIR/companies.todo"
}
trap on_exit INT TERM

# Snapshot (first run or --refresh)
ensure_snapshot Company   gmr_id        "$STATE_DIR/companies.todo"
ensure_snapshot Authority authority_id  "$STATE_DIR/authorities.todo"

# Drain
drain_file Company   "$STATE_DIR/companies.todo"
drain_file Authority "$STATE_DIR/authorities.todo"

log "full scan complete"
