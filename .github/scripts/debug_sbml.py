"""
Diagnostic script for the libRoadRunner CI failure: bionetgen's own
exception message only says "SBML generation did not produce
temp_sbml.xml" without surfacing BNG2.pl's actual stderr/reasoning.

This reproduces the exact call that failed
(bionetgen.bngmodel(...).setup_simulator(), the same path
TestSetupAndRunByPropagator::test_setup_and_run[libRoadRunner] exercises)
with logging turned up to DEBUG, to see what BNG2.pl itself is actually
complaining about.

Run manually with: python .github/scripts/debug_sbml.py
Remove this script (and its CI step) once the root cause is found and
fixed.
"""
import logging
logging.basicConfig(level=logging.DEBUG)

import os
import bionetgen

bng_path = os.path.join(os.path.dirname(bionetgen.__file__), "bng-linux")
print("bng_path:", bng_path, "exists:", os.path.isdir(bng_path))

model = bionetgen.bngmodel("tests/test.bngl")
try:
    model.setup_simulator()
    print("setup_simulator() succeeded (unexpected given the pytest failure)")
except Exception as e:
    print("setup_simulator() raised:", type(e).__name__, str(e))
    cause = e.__cause__
    depth = 0
    while cause is not None and depth < 5:
        print("  caused by:", type(cause).__name__, str(cause))
        cause = getattr(cause, "__cause__", None)
        depth += 1