# Settings for the 2023 run with the browseGP velocity prior, velocity parameters FITTED on the sample tiles.
# USE:  source env_2023_browsegp.sh      (source it, do not run it with bash, or the exports are lost)  then:  bash submit_test.sh [pilot]
# The sbatch jobs inherit these variables from the shell you submit from (--export=ALL).

export BSST_TEST_FILE=/group/oi019/wedge/data/browseSST/test_bb23_2023.nc
export BSST_RUN=/group/oi019/wedge/data/browseSST/run_bg_free_full     # use a NEW directory for each configuration
export BSST_PRIOR=browsegp
export BSST_BLOCKS=rot,div,fast
export BSST_FREE_VEL=rot,div,fast       # set to empty (export BSST_FREE_VEL=) for the control run with browseGP's velocity values held fixed
export BSST_FIT_HOURS=72                # fit each sample tile on its central 72 h
