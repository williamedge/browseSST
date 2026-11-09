"""
HPC test driver for the 3 deg x 3 deg x 28 d real-data test (same style as browseGP/scripts/run_*.py).

  python run_test.py <stage> [pilot] [dry]

stages (run in this order; sbat_test_<stage>.sbatch calls this, `submit_test.sh` chains them):
  manifest  tile layout of the test file                                 (seconds; also run on a login node)
  tide      regional barotropic tide M2 S2 (day-by-day heating nuisance) on the whole test cube   (pilot: coarser time stride; not needed with BSST_TIDE_SOURCE=cube)
  hyper     hyperparameter fit on one sample tile per array task          (pilot: 3 + 3 iterations)
  pool      pool the per-tile fits into hyper.json
  tiles     one tile per array task                                       (pilot: ONLY the first sample tile, to time it and read its memory with sacct)
  merge     blend the tile files into merged.nc
`pilot` = quick sanity run; `dry` = print the command without running it. Paths below can be overridden by the environment variables BSST_ROOT, BSST_TEST_FILE, BSST_RUN (and the tile size by
BSST_CORE='168,28,28', BSST_BUFFER='12,4,4', BSST_TIDE_SOURCE=fit|cube|none), which the local tests use; the defaults are the HPC locations (EDIT if the repository / data live elsewhere).
"""
import concurrent.futures
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ARGS = sys.argv[1:]
STAGE = ARGS[0] if ARGS else ''
PILOT = 'pilot' in ARGS[1:]
DRY = 'dry' in ARGS[1:]
NCORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))   # keep in sync with --cpus-per-task in sbat_test_*.sbatch

ROOT = Path(os.environ.get('BSST_ROOT', '/group/oi019/wedge/pl/pckgs/browseSST'))
TEST_FILE = os.environ.get('BSST_TEST_FILE', '/group/oi019/wedge/data/browseSST/test_bb23_may2026.nc')
RUN = Path(os.environ.get('BSST_RUN', '/group/oi019/wedge/data/browseSST/run_test_bb23'))
HPC = ROOT / 'hpc'
CORE = os.environ.get('BSST_CORE', '168,28,28').split(',')      # tile core: hours, lat cells, lon cells (1 week x 1 deg x 1 deg at 4 km)
BUFFER = os.environ.get('BSST_BUFFER', '12,4,4').split(',')     # buffer on the sides that border another tile
MIN_DEPTH = '100'                 # m; the test file already has depths < 100 m set to NaN
PRIOR = os.environ.get('BSST_PRIOR', 'helmholtz')              # 'browsegp': stage-2 velocity prior = browseGP's in-situ model (blocks BSST_BLOCKS), its velocity parameters FITTED on the sample tiles for the blocks in BSST_FREE_VEL
BLOCKS = os.environ.get('BSST_BLOCKS', 'rot,div,fast')
FREE_VEL = os.environ.get('BSST_FREE_VEL', 'rot,div,fast')     # '' = keep browseGP's values fixed
FIT_HOURS = os.environ.get('BSST_FIT_HOURS', '0')              # >0: hyper fit on the central N hours of the sample tile (cost is linear in points)
TIDE_SOURCE = os.environ.get('BSST_TIDE_SOURCE', 'fit')       # 'fit' = regional SST regression (stage `tide`); 'cube' = external u_tide / v_tide variables in the test file (e.g. FES2022); 'none'
version = f'{STAGE}_pilot' if PILOT else STAGE


def tide_args():
    if TIDE_SOURCE == 'cube':
        return ['--tide-from-cube']
    if TIDE_SOURCE == 'none':
        return []
    return ['--tide', str(RUN / 'tide' / 'tide_model.pkl')]


def prior_args(fit=False):
    if PRIOR != 'browsegp':
        return []
    a = ['--prior', 'browsegp', '--blocks', BLOCKS]
    return a + (['--free-vel', FREE_VEL] if fit and FREE_VEL else [])


def sample_tiles():
    return (RUN / 'manifest_sample_tiles.txt').read_text().strip()


def commands():
    py = ['python', '-u']
    k = os.environ.get('SLURM_ARRAY_TASK_ID')
    if STAGE == 'manifest':
        return py + [str(HPC / 'make_manifest.py'), '--cube', TEST_FILE, '--core', *CORE, '--buffer', *BUFFER, '--out', str(RUN / 'manifest.json'), '--n-sample', '4']
    if STAGE == 'tide':
        return py + [str(HPC / 'run_tide.py'), '--cube', TEST_FILE, '--out-dir', str(RUN / 'tide'), '--min-depth', MIN_DEPTH, '--time-stride', '6' if PILOT else '2', '--n-jobs', str(NCORES), '--nuisance-t-days', '1', '--spacing', '150e3']
    if STAGE == 'hyper':
        it = '3' if PILOT else '25'
        tiles = sample_tiles()
        idx = ['--array-index', k] if k is not None else ['--tile', tiles.split(',')[0]]
        return py + [str(HPC / 'fit_hyper.py'), '--cube', TEST_FILE, '--manifest', str(RUN / 'manifest.json'), '--tiles', tiles, *idx, *tide_args(),
                     '--hyper-in', str(HPC / ('hyper_browsegp_start.json' if PRIOR == 'browsegp' else 'hyper_real_start.json')), '--out-dir', str(RUN / 'hyper'), '--maxiter1', it, '--maxiter2', it, '--noise-mode', 'mle',
                     '--fit-hours', FIT_HOURS, *prior_args(fit=True)]
    if STAGE == 'pool':
        return py + [str(HPC / 'pool_hyper.py'), '--inputs', *sorted(glob.glob(str(RUN / 'hyper' / 'hyper_*.json'))), '--out', str(RUN / 'hyper.json')]
    if STAGE == 'tiles':
        hyper = RUN / 'hyper.json' if (RUN / 'hyper.json').exists() else HPC / ('hyper_browsegp_start.json' if PRIOR == 'browsegp' else 'hyper_real_start.json')
        sel = ['--tile', sample_tiles().split(',')[0]] if (PILOT or k is None) else ['--array-index', k]
        return py + [str(HPC / 'run_tile.py'), '--cube', TEST_FILE, '--manifest', str(RUN / 'manifest.json'), '--hyper', str(hyper), *tide_args(),
                     '--out-dir', str(RUN / ('tiles_pilot' if PILOT else 'tiles')), *sel, *prior_args()]
    if STAGE == 'merge':
        return py + [str(HPC / 'merge.py'), '--cube', TEST_FILE, '--manifest', str(RUN / 'manifest.json'), '--tiles-dir', str(RUN / 'tiles'), '--out', str(RUN / 'merged.nc')]
    sys.exit(f'unknown stage {STAGE!r}; use manifest | tide | hyper | pool | tiles | merge')


def run_job():
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'logs').mkdir(exist_ok=True)
    cmd = commands()
    tag = f"{version}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}".rstrip('_')
    log_filename = RUN / 'logs' / f'log_{tag}.txt'
    if DRY:
        print(' '.join(cmd))
        return 0
    with open(log_filename, 'w') as log_file:
        log_file.write(f'Job started at {datetime.now()}\n')
        log_file.write(f'Command: {cmd}\n')
        print(f'Running stage={version}', flush=True)
        proc = subprocess.run(cmd, stdout=log_file, stderr=log_file, cwd=str(HPC))
        log_file.write(f'\nJob finished at {datetime.now()}\n')
        log_file.write(f'Return code: {proc.returncode}\n')
    return proc.returncode


with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    future = executor.submit(run_job)
    rc = future.result()
    print(f'Job finished with return code {rc}', flush=True)
    sys.exit(rc)
