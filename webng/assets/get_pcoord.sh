#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi
cd $WEST_SIM_ROOT
tail -n 1 bngl_conf/init.gdat > $WEST_PCOORD_RETURN
if [ -n "$SEG_DEBUG" ] ; then
  head -v $WEST_PCOORD_RETURN
fi