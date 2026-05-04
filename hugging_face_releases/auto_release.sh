#!/usr/bin/env bash
#
# Automated HuggingFace model release pipeline.
#
# Parses the training command to extract architecture parameters,
# converts .pt to safetensors, packages the model, and verifies
# that the HF model produces identical outputs.
#
# Usage:
#   bash hugging_face_releases/auto_release.sh \
#       --command "python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 ..." \
#       --checkpoint logs/segmentation/wandb/latest-run/files/best.pt
#
#   Or point to a script file:
#   bash hugging_face_releases/auto_release.sh \
#       --command_file temp/run_segmentation_pasvoc.sh \
#       --checkpoint logs/segmentation/wandb/latest-run/files/best.pt

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
COMMAND=""
COMMAND_FILE=""
CHECKPOINT=""
OUTPUT_BASE="hugging_face_releases"
SKIP_TEST=false

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Required (one of --command or --command_file):
  --command "python seg_main.py ..."   Training command string
  --command_file FILE                  File containing the training command
  --checkpoint PATH                   Path to the .pt checkpoint

Optional:
  --output_base DIR     Base directory for output (default: hugging_face_releases)
  --skip_test           Skip the verification step
  -h, --help            Show this help
EOF
    exit 1
}

# ── Parse script arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --command)      COMMAND="$2"; shift 2 ;;
        --command_file) COMMAND_FILE="$2"; shift 2 ;;
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --output_base)  OUTPUT_BASE="$2"; shift 2 ;;
        --skip_test)    SKIP_TEST=true; shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Load command from file if needed ─────────────────────────────────────────
if [[ -z "$COMMAND" && -n "$COMMAND_FILE" ]]; then
    if [[ ! -f "$COMMAND_FILE" ]]; then
        echo "ERROR: command file not found: $COMMAND_FILE"; exit 1
    fi
    COMMAND=$(grep -m1 'python.*main.*\.py' "$COMMAND_FILE" || true)
    if [[ -z "$COMMAND" ]]; then
        echo "ERROR: no python training command found in $COMMAND_FILE"; exit 1
    fi
fi

if [[ -z "$COMMAND" ]]; then
    echo "ERROR: --command or --command_file is required"; usage
fi
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required"; usage
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1
fi

# ── Helper: extract a flag value from the training command ───────────────────
get_val() {
    # $1 = flag name (e.g. --soft_thresholding)
    # Prints the value or empty string if not found
    echo "$COMMAND" | grep -oP "(?<=$1\s)\S+" || true
}

has_flag() {
    # $1 = flag name (e.g. --hard_mask)
    echo "$COMMAND" | grep -qw -- "$1" && echo "true" || echo "false"
}

# ── Extract parameters from training command ─────────────────────────────────
CONFIG=$(get_val "--config")
CONFIG_NAME=$(get_val "--config_name")
SOFT_THRESH=$(get_val "--soft_thresholding")
SOFT_THRESH_POS=$(get_val "--soft_thresholding_pos")
N_ROTATIONS=$(get_val "--n_rotations")
HARD_MASK=$(has_flag "--hard_mask")
PRESERVE_NORM=$(has_flag "--preserve_norm")
GROUP_TYPE=$(get_val "--group_type")
JOINT_DECOMP=$(has_flag "--joint_decomposition")

# Defaults for values not in the command
: "${CONFIG:=config/segmentation.yaml}"
: "${CONFIG_NAME:=segformer_base}"
: "${SOFT_THRESH:=0.0}"
: "${SOFT_THRESH_POS:=0.0}"
: "${N_ROTATIONS:=4}"
: "${GROUP_TYPE:=rotation}"

# ── Resolve model type and backbone from config YAML ─────────────────────────
resolve_config_field() {
    # $1 = section ("model" or "data"), $2 = field name
    python3 -c "
import sys
sys.path.insert(0, '.')
from config.utils import load_config
cfg = load_config('${CONFIG}', config_name='${CONFIG_NAME}')
print(cfg.get('$1', {}).get('$2', ''))
"
}

MODEL_TYPE=$(resolve_config_field "model" "type")
PRETRAINED_MODEL=$(resolve_config_field "model" "pretrained_model")
NUM_LABELS=$(resolve_config_field "model" "num_labels")
DATASET=$(resolve_config_field "data" "dataset")
DECOMP_METHOD=$(resolve_config_field "model" "decomposition_method")
: "${DECOMP_METHOD:=schur}"

if [[ -z "$MODEL_TYPE" ]]; then
    echo "ERROR: could not resolve model.type from config ${CONFIG} [${CONFIG_NAME}]"; exit 1
fi
if [[ -z "$NUM_LABELS" ]]; then
    echo "ERROR: could not resolve model.num_labels from config"; exit 1
fi

# Map training model type -> HF model_arch
# Note: filtered_segformer is not yet supported for HF release (no modeling file in _shared/).
declare -A TYPE_TO_ARCH=(
    ["filtered_vit"]="filtered_vit"
    ["filtered_dinov2"]="filtered_dinov2"
    ["filtered_vit_seg"]="filtered_vit_seg"
    ["filtered_dino2_seg"]="filtered_dino2_seg"
)

MODEL_ARCH="${TYPE_TO_ARCH[$MODEL_TYPE]:-}"
if [[ -z "$MODEL_ARCH" ]]; then
    echo "ERROR: unsupported model type '$MODEL_TYPE' for HF release."
    echo "       Supported types: ${!TYPE_TO_ARCH[*]}"
    echo "       Note: 'filtered_segformer' is not yet supported (no modeling file in _shared/)."
    exit 1
fi

# ── Build output folder name ─────────────────────────────────────────────────
# Pattern: filtered-{backbone_short}-{dataset_tag}-seg-c{n_rot}-s{soft}
BACKBONE_SHORT=$(echo "$PRETRAINED_MODEL" | sed 's|.*/||')  # e.g. "vit-base-patch16-224"
if echo "$MODEL_TYPE" | grep -q "seg"; then
    # Derive a short dataset tag from the data.dataset config field.
    case "$DATASET" in
        pascal_voc*)  DATASET_TAG="voc" ;;
        ade20k*)      DATASET_TAG="ade" ;;
        *)            DATASET_TAG="${DATASET:-voc}" ;;  # fallback to raw name or "voc"
    esac
    TASK_TAG="${DATASET_TAG}-seg"
else
    TASK_TAG="imagenet"
fi
FOLDER_NAME="filtered-${BACKBONE_SHORT}-${TASK_TAG}-c${N_ROTATIONS}-s${SOFT_THRESH_POS}"
OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER_NAME}"
SAFETENSORS_DIR="temp/converted_${FOLDER_NAME}"

# ── Print summary ────────────────────────────────────────────────────────────
echo "===================================================================="
echo "  Automated HuggingFace Release Pipeline"
echo "===================================================================="
echo "  Config         : ${CONFIG} [${CONFIG_NAME}]"
echo "  Dataset        : ${DATASET:-unknown}"
echo "  Model type     : ${MODEL_TYPE}"
echo "  Model arch     : ${MODEL_ARCH}"
echo "  Backbone       : ${PRETRAINED_MODEL}"
echo "  Num labels     : ${NUM_LABELS}"
echo "  N rotations    : ${N_ROTATIONS}"
echo "  Soft threshold : ${SOFT_THRESH}"
echo "  Soft thresh pos: ${SOFT_THRESH_POS}"
echo "  Hard mask      : ${HARD_MASK}"
echo "  Group type     : ${GROUP_TYPE}"
echo "  Checkpoint     : ${CHECKPOINT}"
echo "  Safetensors dir: ${SAFETENSORS_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "===================================================================="

# ── Step 1: Convert .pt -> safetensors ───────────────────────────────────────
echo ""
echo "Step 1/3: Converting .pt to safetensors..."
python hugging_face_releases/convert_to_safetensors.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$SAFETENSORS_DIR" \
    --verify

# ── Step 2: Package model ────────────────────────────────────────────────────
echo ""
echo "Step 2/3: Packaging model for HuggingFace..."
PACKAGE_CMD=(
    python hugging_face_releases/package_model.py
    --safetensors "${SAFETENSORS_DIR}/model.safetensors"
    --output_dir "$OUTPUT_DIR"
    --model_arch "$MODEL_ARCH"
    --pretrained_model "$PRETRAINED_MODEL"
    --num_labels "$NUM_LABELS"
    --n_rotations "$N_ROTATIONS"
    --soft_thresholding "$SOFT_THRESH"
    --soft_thresholding_pos "$SOFT_THRESH_POS"
    --group_type "$GROUP_TYPE"
    --decomposition_method "$DECOMP_METHOD"
)
[[ "$HARD_MASK" == "true" ]]    && PACKAGE_CMD+=(--hard_mask)
[[ "$PRESERVE_NORM" == "true" ]] && PACKAGE_CMD+=(--preserve_norm)
[[ "$JOINT_DECOMP" == "true" ]] && PACKAGE_CMD+=(--joint_decomposition)

"${PACKAGE_CMD[@]}"

# ── Step 3: Verify ───────────────────────────────────────────────────────────
if [[ "$SKIP_TEST" == "true" ]]; then
    echo ""
    echo "Step 3/3: SKIPPED (--skip_test)"
else
    echo ""
    echo "Step 3/3: Verifying HF model matches original..."
    TEST_CMD=(
        python hugging_face_releases/test_release.py
        --config "$CONFIG"
        --config_name "$CONFIG_NAME"
        --checkpoint "$CHECKPOINT"
        --hf_dir "$OUTPUT_DIR"
        --n_rotations "$N_ROTATIONS"
        --soft_thresholding "$SOFT_THRESH"
        --soft_thresholding_pos "$SOFT_THRESH_POS"
    )
    [[ "$HARD_MASK" == "true" ]] && TEST_CMD+=(--hard_mask)

    "${TEST_CMD[@]}"
fi

echo ""
echo "===================================================================="
echo "  Release pipeline complete!"
echo "  Model folder: ${OUTPUT_DIR}"
echo ""
echo "  To upload:"
echo "    huggingface-cli upload <username>/<repo> ${OUTPUT_DIR}"
echo "===================================================================="
