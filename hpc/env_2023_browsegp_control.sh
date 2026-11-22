# CONTROL run: the same chain as env_2023_browsegp.sh but with browseGP's v0.08 velocity parameters HELD FIXED (nothing velocity-related is fitted).
# Only stage 1, the source term S, the diurnal term and the stage-2 noise are fitted on the 4 sample tiles.
# USE:  source env_2023_browsegp_control.sh   then:  mkdir -p $BSST_RUN && cp -r <earlier run>/tide $BSST_RUN/   then:  bash submit_test.sh from-hyper
#       (without the copy, use `bash submit_test.sh` and the tide stage is re-run: same result, 1 h more)

export BSST_TEST_FILE=/group/oi019/wedge/data/browseSST/test_bb23_2023.nc
export BSST_RUN=/group/oi019/wedge/data/browseSST/run_v008_control     # a NEW directory
export BSST_PRIOR=browsegp
export BSST_BLOCKS=rot,div,d2
export BSST_DIV_KERNEL=ring
export BSST_FREE_VEL=                   # EMPTY: velocity parameters stay at browseGP v0.08 (hpc/browsegp_vel_v008.json)
export BSST_FIX_VEL=ell_div,tau_div     # irrelevant while FREE_VEL is empty
export BSST_FIT_HOURS=72
export BSST_MIN_DEPTH=250
