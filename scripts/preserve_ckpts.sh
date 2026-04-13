#!/bin/bash
# Hardlink new global_step_* dirs into a sibling _preserved/ dir so they
# survive verl's max_actor_ckpt_to_keep rotation.
#
# Uses `cp -aln` (no-clobber hardlink) so it is safe to re-run — missing
# files get filled in, existing hardlinks stay. This handles the race
# where verl writes model shards first and config/extra_state later.
CKPT_DIR="${1:?usage: $0 <ckpt_dir>}"
BACKUP_DIR="${CKPT_DIR}_preserved"
mkdir -p "$BACKUP_DIR"
echo "[watch] $CKPT_DIR -> $BACKUP_DIR"
while true; do
  for d in "$CKPT_DIR"/global_step_*; do
    [ -d "$d" ] || continue
    [ -d "$d/actor" ] || continue
    [ -n "$(ls -A "$d/actor" 2>/dev/null)" ] || continue
    name=$(basename "$d")
    mkdir -p "$BACKUP_DIR/$name/actor"
    cp -aln "$d/actor/." "$BACKUP_DIR/$name/actor/" 2>/dev/null
  done
  sleep 30
done
