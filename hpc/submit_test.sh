#!/bin/bash
# Chain the real-data test on the HPC:  manifest -> tide -> hyper (array) -> pool -> tiles (array) -> merge.
#   bash submit_test.sh pilot   : manifest, then ONE pilot tile (and a pilot tide fit); read the time / memory with `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State`
#   bash submit_test.sh         : the full chain (run the pilot first and size --mem / --time in the sbat_test_*.sbatch files from it)
# Paths: the same constants as run_test.py (edit there or set BSST_ROOT / BSST_TEST_FILE / BSST_RUN).
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-full}
RUN=${BSST_RUN:-/group/oi019/wedge/data/browseSST/run_test_bb23}
python run_test.py manifest                                   # quick: tile layout + sample tiles (login node is fine)
NTILES=$(python -c "import json;print(len(json.load(open('$RUN/manifest.json'))['tiles']))")
NSAMP=$(python -c "print(len(open('$RUN/manifest_sample_tiles.txt').read().strip().split(',')))")
echo "manifest: $NTILES tiles, $NSAMP sample tiles"
if [ "$MODE" = "pilot" ]; then
  J1=$(sbatch --parsable sbat_test_tide.sbatch pilot)
  J2=$(sbatch --parsable --dependency=afterok:$J1 sbat_test_tiles.sbatch pilot)
  echo "pilot submitted: tide $J1, tile $J2   ->   sacct -j $J2 --format=JobID,Elapsed,MaxRSS,State"
  exit 0
fi
J1=$(sbatch --parsable sbat_test_tide.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 --array=0-$((NSAMP-1)) sbat_test_hyper.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J2 sbat_test_pool.sbatch)
J4=$(sbatch --parsable --dependency=afterok:$J3 --array=0-$((NTILES-1))%8 sbat_test_tiles.sbatch)
J5=$(sbatch --parsable --dependency=afterok:$J4 sbat_test_merge.sbatch)
echo "submitted: tide $J1, hyper $J2, pool $J3, tiles $J4 (array 0-$((NTILES-1)), 8 at a time), merge $J5"
