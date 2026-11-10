#!/usr/bin/env python
import numpy, os, sys
def pcoord_loader(fieldname, coord_filename, segment, single_point=False):
    pcoord = numpy.loadtxt(coord_filename, dtype = numpy.float32)
    raw_shape = pcoord.shape
    # numpy.loadtxt silently collapses to a 1D array whenever the
    # file has exactly one row OR exactly one column -- don't
    # assume 2D just because a full trajectory was requested (or
    # 1D just because a single point was). Force 2D explicitly,
    # then slice consistently regardless of how many rows loaded.
    pcoord = numpy.atleast_2d(pcoord)
    if not single_point:
        segment.pcoord = pcoord[:, 1:]
    else:
        segment.pcoord = pcoord[-1, 1:]
    # DEBUG: every shell-level check upstream (run_network exit code,
    # seg.gdat non-empty, seg.gdat has non-comment data rows) has
    # passed, yet segments still end up with an empty pcoord -- this
    # traces exactly what this function sees and produces.
    # NOTE: unconditional for now -- pcoord_loader runs inside the
    # parent w_run process itself (it's a Python callable, not a
    # subprocess), so SEG_DEBUG/PROPAGATION_DEBUG (which west.cfg only
    # injects into spawned child processes like runseg.sh) are not
    # actually set in this process's os.environ -- gating on them
    # here silently produced zero output. Remove this print once the
    # root cause is found.
    if True:
        sys.stderr.write(
            'pcoord_loader: coord_filename={} single_point={} '            'raw_loadtxt_shape={} after_atleast_2d_shape={} '            'final_pcoord_shape={}\n'.format(
                coord_filename, single_point, raw_shape, pcoord.shape,
                segment.pcoord.shape
            )
        )
        try:
            with open(coord_filename) as fh:
                content = fh.read()
            sys.stderr.write(
                '--- {} contents ---\n{}\n--- end {} ---\n'.format(
                    coord_filename, content, coord_filename
                )
            )
        except Exception as e:
            sys.stderr.write(
                'could not read {} for debug: {}\n'.format(coord_filename, e)
            )