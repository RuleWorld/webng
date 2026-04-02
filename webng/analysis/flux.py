import os, h5py
import subprocess as sbpc
import matplotlib.pyplot as plt
import numpy as np
from webng.analysis.analysis import weAnalysis

import warnings
warnings.filterwarnings("ignore")


class weFlux(weAnalysis):
    """
    Flux and rate analysis for WESTPA simulations.

    Calls w_fluxanl to compute the flux evolution into the target state,
    then prints the average flux and rate with confidence intervals and
    plots the rate evolution over iterations with shaded confidence intervals.

    All values are reported in units of inverse tau (tau^-1).
    Works with any binning scheme as long as a target state is defined.
    """

    def __init__(self, opts):
        super().__init__(opts)
        self.h5file_path = os.path.join("..", "west.h5")
        self.h5file = h5py.File(self.h5file_path, "r")
        self.first_iter = self._getd(opts, "first-iter", default=None, required=False)
        self.last_iter  = self._getd(opts, "last-iter",  default=None, required=False)
        self.first_iter, self.last_iter = self.set_iter_range(
            self.first_iter, self.last_iter
        )
        self.step_iter = self._getd(opts, "step-iter", default=1,   required=False)
        self.output    = self._getd(opts, "output",    default="flux.png", required=False)

    def set_iter_range(self, first_iter, last_iter):
        if first_iter is None:
            first_iter = 1
        if last_iter is None:
            last_iter = self.h5file.attrs["west_current_iteration"] - 1
        return first_iter, last_iter

    def run(self):
        # --- Run w_fluxanl if output file does not exist ---
        if not os.path.isfile("fluxanl.h5"):
            print("fluxanl.h5 does not exist. Running w_fluxanl")
            command = [
                "w_fluxanl",
                "-W",           "{}".format(self.h5file_path),
                "--first-iter", "{}".format(self.first_iter),
                "--last-iter",  "{}".format(self.last_iter),
                "--evol",
                "--evol-step",  "{}".format(self.step_iter),
                "-o",           "fluxanl.h5",
            ]
            proc = sbpc.Popen(command)
            proc.wait()

        # --- Read results ---
        with h5py.File("fluxanl.h5", "r") as f:
            # Only keep target_N groups, sorted numerically
            target_keys = sorted(
                [k for k in f["target_flux"].keys() if k.startswith("target_")],
                key=lambda k: int(k.split("_")[1])
            )
            n_targets = len(target_keys)

            # avg_flux has one row per target
            avg_flux    = float(sum(f["avg_flux"]["expected"][i]  for i in range(n_targets)))
            avg_ci_low  = float(sum(f["avg_flux"]["ci_lbound"][i] for i in range(n_targets)))
            avg_ci_high = float(sum(f["avg_flux"]["ci_ubound"][i] for i in range(n_targets)))

            # Read evolution data — accumulate across all targets
            evol_mean  = None
            evol_low   = None
            evol_high  = None
            iter_start = None
            iter_stop  = None

            for i, key in enumerate(target_keys):
                ds  = f["target_flux"][key]["flux_evolution"][:]
                exp = np.array(ds["expected"],  dtype=np.float64)
                lo  = np.array(ds["ci_lbound"], dtype=np.float64)
                hi  = np.array(ds["ci_ubound"], dtype=np.float64)
                if i == 0:
                    evol_mean  = exp
                    evol_low   = lo
                    evol_high  = hi
                    iter_start = np.array(ds["iter_start"], dtype=np.uint32)
                    iter_stop  = np.array(ds["iter_stop"],  dtype=np.uint32)
                else:
                    evol_mean += exp
                    evol_low  += lo
                    evol_high += hi

        # Use iter_stop as x-axis — "rate estimated up to this iteration"
        x_axis = iter_stop

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