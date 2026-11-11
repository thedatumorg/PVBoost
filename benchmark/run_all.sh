#!/usr/bin/env bash
# Run PVBoost for the paper's 16 source detectors x 3 seeds on TSB-AD-U and TSB-AD-M, then summarize.
#
#   TSBAD_ROOT=/path/to/TSB-AD SCORE_ROOT=/path/to/scores bash benchmark/run_all.sh
#
# SCORE_ROOT must contain uni/<detector>/<series>.npy and multi/<detector>/<series>.npy
# (see README, "Source scores"). Jobs run sequentially; set JOBS>1 to run that many at once.
set -euo pipefail
cd "$(dirname "$0")/.."

TSBAD_ROOT=${TSBAD_ROOT:?set TSBAD_ROOT to your TSB-AD checkout}
SCORE_ROOT=${SCORE_ROOT:?set SCORE_ROOT to the precomputed source scores}
SAVE_DIR=${SAVE_DIR:-results}
SEEDS=${SEEDS:-"0 1 2"}
BENCHMARKS=${BENCHMARKS:-"U M"}
JOBS=${JOBS:-1}

# Score-directory names as written by TSB-AD's Run_Detector_{U,M}.py. On TSB-AD-U the
# statistical detectors are TSB-AD's subsequence variants (Sub_*, KMeansAD_U).
U_DETECTORS="Sub_LOF Sub_KNN KMeansAD_U Sub_OCSVM Sub_HBOS Sub_IForest Sub_PCA LSTMAD PatchTST \
AutoEncoder OmniAnomaly TimesNet AnomalyTransformer TranAD Chronos2 Toto2"
M_DETECTORS="LOF KNN KMeansAD OCSVM HBOS IForest PCA LSTMAD PatchTST \
AutoEncoder OmniAnomaly TimesNet AnomalyTransformer TranAD Chronos2 Toto2"

export DATA="${TSBAD_ROOT}/Datasets" SCORE_ROOT SAVE_DIR
run_one() {  # $1 benchmark, $2 detector, $3 seed
  local sub=uni; [ "$1" = M ] && sub=multi
  python benchmark/run_tsbad.py --benchmark "$1" --detector "$2" --seed "$3" \
    --dataset_dir "${DATA}/TSB-AD-$1" --score_dir "${SCORE_ROOT}/${sub}" \
    --file_list "${DATA}/File_List/TSB-AD-$1-Eva.csv" --save_dir "${SAVE_DIR}" \
    > "${SAVE_DIR}/logs/$1_$2_seed$3.log" 2>&1 && echo "done  $1 $2 seed$3" || echo "FAILED $1 $2 seed$3 (see ${SAVE_DIR}/logs/$1_$2_seed$3.log)"
}
export -f run_one

mkdir -p "${SAVE_DIR}/logs"
for b in $BENCHMARKS; do
  if [ "$b" = U ]; then dets=$U_DETECTORS; sub=uni; else dets=$M_DETECTORS; sub=multi; fi
  for d in $dets; do
    [ -d "${SCORE_ROOT}/${sub}/${d}" ] || { echo "missing scores: ${SCORE_ROOT}/${sub}/${d}" >&2; exit 1; }
    for s in $SEEDS; do echo "$b $d $s"; done
  done
done | xargs -P "$JOBS" -n 3 bash -c 'run_one "$@"' _

python benchmark/summarize.py --save_dir "$SAVE_DIR"
