#!/bin/bash -l
#SBATCH --job-name=oi019_wedge_browseSST_merge
#SBATCH --output=browseSST_merge_%j.out
#SBATCH --error=browseSST_merge_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=work
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=00099894@uwa.edu.au

source activate /group/oi019/wedge/env/drift

cd /group/oi019/wedge/pl/pckgs/browseSST/hpc

python run_test.py merge "$@"
