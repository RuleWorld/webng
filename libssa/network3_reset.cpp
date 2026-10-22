// network3_reset.cpp
//
// Compiles Network3's network.cpp as part of this file (build_libssa.sh leaves
// network.cpp out of its normal source list and compiles this instead), so that
// libssa can reach Network3's file-local Gillespie state (`static struct {...} GSP`)
// and reset it.
//
// Why this is needed: Network3 was written for run_network, which does one
// simulation per process. Its init functions add to global state without clearing
// it first -- GSP.a_tot (total propensity) is summed with +=, GSP.prop and
// network.spec_groups_vec are appended with push_back, functions and their
// dependency lists are appended, and nested dependency vectors are resize()d,
// which keeps the old inner lists. So a second ssa_init in one process inherited
// the first one's state; the inflated total propensity then sent reaction
// selection past the end of the reaction array (segfault), and anything that
// didn't crash was running on stale dependencies.
//
// Resetting both globals to value-initialized (all-zero, empty) state returns
// Network3 to exactly where it is when the library is first loaded.
#include "network.cpp"

void libssa_reset_network3() {
    if (network.name) free(network.name);
    network = NETWORK();
    GSP = decltype(GSP)();
}

// libssa always simulates a complete network that BNG2.pl generated ahead of
// time. Network3 also supports on-the-fly generation: when a species with no
// reactions of its own gets its first molecule, it prints "edgepop:" and waits
// on stdin for BNG2.pl to send new reactions. Inside libssa nothing ever
// answers, so the simulation hung forever (or crashed on a closed stdin) for any
// model with such a species starting at zero, e.g. a terminal product.
// ever_populated only drives that check (and a statistic run_network writes to
// its own output files), so marking every species as already populated turns
// on-the-fly generation off without changing anything else.
void libssa_disable_onthefly_generation() {
    for (int i = 0; i < GSP.nc; ++i)
        GSP.ever_populated[i] = 1;
}
