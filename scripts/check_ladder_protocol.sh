#!/bin/bash
# v1.153 (2026-09-24): CPU gate for the fixed ladder protocol (SC_LADDER_PROTO=2) before any GPU time is spent.
# Runs the REAL eval_flip_ll.py with the stub model (scripts/dryrun_eval.py) on scratch copies of the E4B COCO-CF runs and
# checks, per variant: the run completes (the in-code audit stops it on the first violation), and an INPUT-INSENSITIVE
# scorer (SC_DRY_CONST=1) gets exactly zero Flip under protocol 2 (and, for contrast, a non-zero Flip under protocol 1).
# Usage: bash scripts/check_ladder_protocol.sh <scratch_dir>
set -u
cd .; module load python/3.12.3 2>/dev/null; source .venv/bin/activate
S=${1:?scratch dir}; mkdir -p "$S"
W=coco_neo_ctrl_1tok_ctm_cp_seed0_gemma4e4b; P=coco_pt_na_1tok_ctm_cp_seed0_gemma4e4b
L=coco_lora_r16_lr2e4_ad_ctm_cp_look_seed0_gemma4e4b; F=coco_base_frozen_0tok_none_seed0_gemma4e4b
for r in $W $P $L; do mkdir -p $S/$r/ckpt/step_800; cp runs/$r/config.json $S/$r/; cp runs/$r/ckpt/step_800/meta.json $S/$r/ckpt/step_800/; done
mkdir -p $S/$F
A="--task color --data_dir data/coco --split test_id --max_n 140 --third_value 1 --out_task coco3 --step 800"
FZ="--frozen_model_id google/gemma-4-E4B-it"
fail=0
run() {   # run <label> <run> <extra args> -- <env...>
  lab=$1; rd=$2; ex=$3; shift 3
  # constant-scorer runs get their own copy of the run dir: they would otherwise overwrite the normal run's file
  case "$lab" in CONST*) rr=c_$rd; mkdir -p $S/$rr; cp -r $S/$rd/config.json $S/$rd/ckpt $S/$rr/ 2>/dev/null ;; *) rr=$rd ;; esac
  touch $S/.t0_$rr
  out=$(env "$@" timeout 1500 python scripts/dryrun_eval.py scripts/eval_flip_ll.py --run_dir $S/$rr $A $ex 2>&1)
  # eval_flip_ll prints no path: take the summary THIS variant wrote (newer than its start), never an earlier one
  f=$(find $S/$rr/eval/coco3_test_id -maxdepth 1 -name "flip_ll*.json" -newer $S/.t0_$rr 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo "FAIL  $lab: $(echo "$out" | grep -E 'audit FAILED|Error|SystemExit|Traceback' | tail -1 | cut -c1-160)"; fail=1; return; fi
  python3 - "$f" "$lab" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); lab = sys.argv[2]; au = d.get("audit_p2", {}); n = au.get("n", 0)
ok_audit = d.get("protocol") != 2 or (n > 0 and all(v == n for k, v in au.items() if k != "n"))
print(f"{'ok  ' if ok_audit else 'FAIL'}  {lab:44s} protocol {d.get('protocol')} n {d.get('n')} audit {'all ' + str(n) if ok_audit else au}"
      f" | look Acc {d['look_acc']:.3f} Flip {d['look_flip']:.3f} Stay {d['look_stay_c']:.3f} | read Acc {d['read_acc']:.3f} Flip {d['read_flip']:.3f} Stay {d['read_stay_c']:.3f}"
      + (f" | distractor {d.get('p2_distractor_rate')}" if d.get("protocol") == 2 else ""))
if "CONST" in lab:
    zero = d["look_flip"] == 0 and d["read_flip"] == 0
    want_zero = d.get("protocol") == 2
    print(f"      constant scorer: Flip {'= 0' if zero else '> 0'} ({'as required' if zero == want_zero else 'UNEXPECTED'})")
    sys.exit(0 if zero == want_zero else 3)
sys.exit(0 if ok_audit else 2)
EOF
  [ $? -eq 0 ] || fail=1
}
run "word, held-out"                 $W "" -- SC_LADDER_PROTO=2 SC_TEMPLATE=heldout
run "word, training template"        $W "" -- SC_LADDER_PROTO=2
run "prefix, held-out"               $P "" -- SC_LADDER_PROTO=2 SC_TEMPLATE=heldout
run "LoRA (image adapter), held-out" $L "" -- SC_LADDER_PROTO=2 SC_TEMPLATE=heldout
run "English (variant 5)"            $F "$FZ" -- SC_LADDER_PROTO=2 INSTR_VARIANT=5
run "no request"                     $F "$FZ" -- SC_LADDER_PROTO=2 NOOP_MODE=1
run "word before caption, held-out"  $W "" -- SC_LADDER_PROTO=2 SC_CAP_FIRST=0 SC_TEMPLATE=heldout
run "word in prefix format, held-out" $W "" -- SC_LADDER_PROTO=2 SC_WORD_PTFMT=1 SC_TEMPLATE=heldout
run "prefix in word slot, held-out"  $P "" -- SC_LADDER_PROTO=2 PT_INSERTED=1 SC_TEMPLATE=heldout
run "prefix instruction, no prefix"  $P "" -- SC_LADDER_PROTO=2 SC_PT_NOPREFIX=1 SC_TEMPLATE=heldout
run "sweep S3, prefix"               $P "" -- SC_LADDER_PROTO=2 SC_SWEEP=S3
run "CONST word, held-out (p2)"      $W "" -- SC_LADDER_PROTO=2 SC_TEMPLATE=heldout SC_DRY_CONST=1
run "CONST prefix, training (p2)"    $P "" -- SC_LADDER_PROTO=2 SC_DRY_CONST=1
run "CONST English (p2)"             $F "$FZ" -- SC_LADDER_PROTO=2 INSTR_VARIANT=5 SC_DRY_CONST=1
run "CONST word, held-out (OLD p1)"  $W "" -- SC_TEMPLATE=heldout SC_DRY_CONST=1
echo "gate: $([ $fail -eq 0 ] && echo PASSED || echo FAILED)"
exit $fail
