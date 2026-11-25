# CONTROL run with the CLOUD-MARGIN MASK and IN-SAMPLE predictions only (see notes, exp 50). Same as env_2023_browsegp_control.sh (browseGP v0.08 velocity parameters held fixed; stage 1, S, diurnal
# and the stage-2 noise are re-fitted on the sample tiles, now on the masked rows) except for the three settings at the bottom.
# USE:  source env_2023_browsegp_control_stencil.sh
#       mkdir -p $BSST_RUN && cp -r /group/oi019/wedge/data/browseSST/run_v008_control/tide $BSST_RUN/
#       bash submit_test.sh from-hyper
#       (without the copy, use `bash submit_test.sh`: the tide stage is re-run, same result, about 1 h more)

source "$(dirname "${BASH_SOURCE[0]}")/env_2023_browsegp_control.sh"
export BSST_RUN=/group/oi019/wedge/data/browseSST/run_v008_control_stencil     # a NEW directory
unset BSST_VEL_JSON                     # the isotropic v0.08 values (hpc/browsegp_vel_v008.json)

export BSST_ROW_MASK=stencil            # stage-2 rows only where SST exists at t-1, t, t+1 and over the +-2 px gradient cross (hyper fit AND tiles)
export BSST_PREDICT_AT=rows             # in-sample only: every output is NaN except at the rows
export BSST_MIN_ROWS=2000               # tiles (and sample tiles) with fewer rows are skipped
