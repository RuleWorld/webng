#!/bin/bash
if [ -n "$SEG_DEBUG" ] ; then
    set -x
    env | sort
fi
cd $WEST_SIM_ROOT || exit 1
if [[ $WEST_CURRENT_ITER -gt 3 ]];then
  PREV_ITER=$(printf "%06d" $((WEST_CURRENT_ITER-3)))
  rm -rf ${WEST_SIM_ROOT}/traj_segs/${PREV_ITER}
  rm -f  seg_logs/${PREV_ITER}-*.log
fi