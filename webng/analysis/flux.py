import os, h5py
import subprocess as sbpc
import matplotlib.pyplot as plt
import numpy as np
import westpa.mclib as mclib
from webng.analysis.analysis import weAnalysis

import warnings
warnings.filterwarnings("ignore")


# w_fluxanl's default: (1 - ALPHA) = 95% confidence intervals
ALPHA = 0.05

# the fields of WESTPA's iter_block_ci_dtype that flux.py uses
EVOL_DTYPE = np.dtype([("iter_start", np.uint32), ("iter_stop", np.uint32),
                       ("expected", np.float64), ("ci_lbound", np.float64),
                       ("ci_ubound", np.float64)])


def flux_ci(fluxes, alpha=ALPHA):
    """
    Mean flux and its (1 - alpha) confidence interval from a per-iteration
    flux series, computed exactly the way w_fluxanl does for one target:
    WESTPA's Monte Carlo bootstrap, accounting for autocorrelation.
    Returns (mean, ci_lbound, ci_ubound).
    """
    avg, lb, ub, _sterr, _correl_len = mclib.mcbs_ci_correl(
        {"dataset": np.asarray(fluxes, dtype=np.float64)},
        estimator=(lambda stride, dataset: np.mean(dataset)),
        alpha=alpha,
        n_sets=mclib.get_bssize(alpha),
        autocorrel_alpha=alpha,
        subsample=np.mean,
        do_correl=True,
        mcbs_enable=True,
    )
    return avg, lb, ub


def flux_evolution(n_iter, fluxes, step, alpha=ALPHA):
    """
    Cumulative evolution of the mean flux and its CI, in blocks of `step`
    iterations, mirroring w_fluxanl's own --evol calculation for one target.
    """
    iter_start, iter_stop = int(n_iter[0]), int(n_iter[-1])
    iter_count = iter_stop - iter_start
    n_blocks = iter_count // step + (1 if iter_count % step > 0 else 0)
    evol = np.empty((n_blocks,), dtype=EVOL_DTYPE)
    for iblock in range(n_blocks):
        istop = min((iblock + 1) * step, len(fluxes))
        avg, lb, ub = flux_ci(fluxes[:istop], alpha)
        evol[iblock] = (iter_start, min(iter_start + (iblock + 1) * step, iter_stop), avg, lb, ub)
    return evol


class weFlux(weAnalysis):
    """
    Flux and rate analysis for WESTPA simulations.

    Calls w_fluxanl to compute the flux evolution into the target state,
    then prints the average flux and rate with confidence intervals and
    plots the rate evolution over iterations with shaded confidence intervals.

    Every line of tstates.txt is treated as part of one sink. With several
    lines (e.g. several bins of one region), their per-iteration fluxes are
    summed and the CI is computed on the total, the same way w_fluxanl
    computes it for a single target.

    All values are reported in units of inverse tau (tau^-1).
    Works with any binning scheme as long as a target state is defined.
    """

    def __init__(self, opts):
        super().__init__(opts)
        self.step_iter = self._getd(opts, "step-iter", default=1,   required=False)
        self.output    = self._getd(opts, "output",    default="flux.png", required=False)

    def run(self):
        # --- Run w_fluxanl if output file does not exist ---
        command = [
            "w_fluxanl",
            "-W",           "{}".format(self.h5file_path),
            "--first-iter", "{}".format(self.first_iter),
            "--last-iter",  "{}".format(self.last_iter),
            "--evol",
            "--evol-step",  "{}".format(self.step_iter),
            "-o",           "fluxanl.h5",
        ]
        stamp = self.stamp("w_fluxanl", first_iter=int(self.first_iter),
                           last_iter=int(self.last_iter), step_iter=self.step_iter)
        self.run_westpa_tool("fluxanl.h5", command, stamp)

        # --- Read results ---
        with h5py.File("fluxanl.h5", "r") as f:
            # Only keep target_N groups, sorted numerically
            target_keys = sorted(
                [k for k in f["target_flux"].keys() if k.startswith("target_")],
                key=lambda k: int(k.split("_")[1])
            )
            n_targets = len(target_keys)
            if n_targets == 1:
                # One target: w_fluxanl's own results, as computed
                avg = f["avg_flux"][0]
                evol = f["target_flux"][target_keys[0]]["flux_evolution"][:]
            else:
                # Every line of tstates.txt is part of ONE sink (e.g. several
                # bins of the same region). w_fluxanl reports each line
                # separately, so add their per-iteration fluxes into one series
                # and compute the CI on that. (Adding the separate CIs' bounds
                # would not give a valid interval for the total.)
                n_iter = f["target_flux"][target_keys[0]]["n_iter"][:]
                for key in target_keys[1:]:
                    if not np.array_equal(f["target_flux"][key]["n_iter"][:], n_iter):
                        raise RuntimeError("{} covers different iterations than {}"
                                           .format(key, target_keys[0]))
                total = sum(np.asarray(f["target_flux"][key]["flux"][:], dtype=np.float64)
                            for key in target_keys)

        if n_targets == 1:
            avg_flux, avg_ci_low, avg_ci_high = (
                float(avg["expected"]), float(avg["ci_lbound"]), float(avg["ci_ubound"]))
        else:
            print("tstates.txt has {} target lines; treating them as one sink "
                  "(per-iteration fluxes summed before computing the CI)".format(n_targets))
            avg_flux, avg_ci_low, avg_ci_high = flux_ci(total)
            evol = flux_evolution(n_iter, total, self.step_iter)
        self.avg_flux = (avg_flux, avg_ci_low, avg_ci_high)

        evol_mean = np.array(evol["expected"],  dtype=np.float64)
        evol_low  = np.array(evol["ci_lbound"], dtype=np.float64)
        evol_high = np.array(evol["ci_ubound"], dtype=np.float64)

        # Use iter_stop as x-axis — "rate estimated up to this iteration"
        x_axis = np.array(evol["iter_stop"], dtype=np.uint32)

        # --- Print average flux and rate ---
        print("\n--- Flux / Rate Analysis ---")
        print("All values in units of tau^-1\n")
        print("Average flux into target state:")
        print("  mean   = {:.6e}".format(avg_flux))
        print("  95% CI = [{:.6e}, {:.6e}]".format(avg_ci_low, avg_ci_high))

        # --- Plot rate evolution ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x_axis, evol_mean, color="steelblue", lw=1.5, label="Mean rate")
        ax.fill_between(
            x_axis, evol_low, evol_high,
            color="steelblue", alpha=0.25, label="95% CI"
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Rate ($\tau^{-1}$)")
        ax.set_title("Rate into target state")
        ax.legend(fontsize=10)

        for kw in ["top", "right"]:
            ax.spines[kw].set_visible(False)

        plt.tight_layout()
        print("\nSaving figure to {}".format(self.output))
        plt.savefig(self.output, dpi=300)

        os.chdir(self.curr_path)
        return fig, ax