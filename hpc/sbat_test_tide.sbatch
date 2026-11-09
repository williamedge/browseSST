#!/bin/bash -l
#SBATCH --job-name=oi019_wedge_browseSST_tide
#SBATCH --output=browseSST_tide_%A_%a.out
#SBATCH --error=browseSST_tide_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --partition=work
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=00099894@uwa.edu.au

source activate /group/oi019/wedge/env/drift

cd /group/oi019/wedge/pl/pckgs/browseSST/hpc

# pass `pilot` for a short sanity run: sbatch sbat_test_tide.sbatch pilot
# array stages (hyper, tiles): sbatch --array=0-N sbat_test_tide.sbatch   (N+1 = number of sample tiles / tiles; see submit_test.sh)
python run_test.py tide "$@"
