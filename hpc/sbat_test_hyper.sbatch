#!/bin/bash -l
#SBATCH --job-name=oi019_wedge_browseSST_hyper
#SBATCH --output=browseSST_hyper_%A_%a.out
#SBATCH --error=browseSST_hyper_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=work
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=00099894@uwa.edu.au

source activate /group/oi019/wedge/env/drift

cd /group/oi019/wedge/pl/pckgs/browseSST/hpc

# pass `pilot` for a short sanity run: sbatch sbat_test_hyper.sbatch pilot
# array stages (hyper, tiles): sbatch --array=0-N sbat_test_hyper.sbatch   (N+1 = number of sample tiles / tiles; see submit_test.sh)
python run_test.py hyper "$@"
