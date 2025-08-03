#!/bin/bash
set -e

# === Configuration ===
#INPUT=input.txt

# prepared with smart_space_remover.py
#INPUT=raw4.txt
# Now we can use same file, bcause space removing will do inside our code
INPUT=./data/mypos-ver.3.0.shuf.notag.nopunc.txt.seg_normalized2

REF=./data/mypos-ver.3.0.shuf.notag.nopunc.txt.seg_normalized2
DICT=./data/myg2p_mypos.dict
SYLFREQ=./data/myMono.freq
ARPA=./data/myMono_clean_syl.arpa
RULES=./data/rules.txt

EXP_DIR=exp_$(date +"%Y%m%d_%H%M")
mkdir -p $EXP_DIR

echo "Created experiment folder: $EXP_DIR"
echo "=============================="

# === List of experimental runs ===
declare -a EXPERIMENTS=(
  "dict_sylfreq"
  "dict_sylfreq_bimmfallback"
  "dict_sylfreq_bimmfallback_bimmboost50"
  "dict_sylfreq_bimmfallback_bimmboost100"
  "dict_sylfreq_arpa_bimmfallback_bimmboost100"
  "dict_only_bimmfallback_bimmboost150"
)

# === Mapping of options ===
declare -A OPTS
OPTS["dict_sylfreq"]="--dict $DICT --sylfreq $SYLFREQ"
OPTS["dict_sylfreq_bimmfallback"]="--dict $DICT --sylfreq $SYLFREQ --use-bimm-fallback"
OPTS["dict_sylfreq_bimmfallback_bimmboost50"]="--dict $DICT --sylfreq $SYLFREQ --use-bimm-fallback --bimm-boost 50"
OPTS["dict_sylfreq_bimmfallback_bimmboost100"]="--dict $DICT --sylfreq $SYLFREQ --use-bimm-fallback --bimm-boost 100"
OPTS["dict_sylfreq_arpa_bimmfallback_bimmboost100"]="--dict $DICT --sylfreq $SYLFREQ --arpa $ARPA --use-bimm-fallback --bimm-boost 100"
OPTS["dict_only_bimmfallback_bimmboost150"]="--dict $DICT --use-bimm-fallback --bimm-boost 150"

# === Run each experiment ===
for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== Running: $EXP ==="

  SEG_FILE="$EXP_DIR/$EXP.seg"
  RESULT_FILE="$EXP_DIR/$EXP.result.txt"
  TOPK_FILE="$EXP_DIR/$EXP.topk.txt"
  #DAG_VIS_DIR="$EXP_DIR/${EXP}_dag_viz"

  # Segment with visualization and space removal
  time python oppa_word.py \
    --input "$INPUT" \
    --output "$SEG_FILE" \
    --postrule-file "$RULES" \
    --space-remove-mode "my_not_num" \
    ${OPTS[$EXP]}

  # Evaluate: simple accuracy
  echo "--- Evaluating tag precision..."
  python2.7 evaluate.py "$SEG_FILE" "$REF" | tee "$RESULT_FILE"

  # Evaluate: Top-K token errors
  echo "--- Evaluating top-k frequent token errors..."
  python eval_segmentation.py -H "$SEG_FILE" -r "$REF" --top-k 50 > "$TOPK_FILE"

  echo "Saved: $SEG_FILE"
  echo "Saved: $RESULT_FILE"
  echo "Saved: $TOPK_FILE"
  echo
done

echo "All experiments completed. Outputs in: $EXP_DIR"
