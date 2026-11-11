import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
from yaml import Loader
from webng.analysis.analysis import weAnalysis
from webng.analysis import markov

# Hacky way to ignore warnings, in particular pyemma insists on Python3
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)


class weNetwork(weAnalysis):
    """
    Rates and mean first passage times (MFPTs) between the states of a
    states.yaml (from `cluster`, with either method, or written by hand).

    One w_assign of the run against states.yaml (assign.h5), then two
    estimates from it:

      direct   w_direct: each walker is labelled with the state it visited
               last (checked at every point of every segment); the rate
               A -> B is the flux of A-labelled weight into B divided by the
               A-labelled probability, and MFPT = tau / rate. Plots
               rate_<A>_to_<B>.png.
      matrix   the bin-to-bin transition matrix at lag tau, from assign.h5:
               each segment's weight counted from the bin it started in to
               the bin it ended in (as w_reweight did; that tool is
               deprecated). Lumped into the states with stationary weights,
               and MFPTs from it: the expected time to first reach B, averaged
               over A's bins weighted by stationary probability (what Tse et
               al. 2018 call transition-matrix mode). Plots mfpt_evolution.png.
               Set `matrix: false` to skip.

    The two answer slightly different questions and need not agree exactly;
    for precise rates of a chosen transition, a run designed for it (targeted,
    history-labelled sampling) is the better estimate. Results are also
    written to network.txt. Times are real time: (WE iterations) * tau.
    """

    def __init__(self, opts):
        # read west.h5 without file locking (set before the base class opens it)
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        super().__init__(opts)
        self.tau = self._getd(opts, "tau", default=10, required=True)
        self.step_iter = self._getd(opts, "step-iter", default=10, required=False)
        self.matrix = bool(self._getd(opts, "matrix", default=True, required=False))

        # the states: those of the assign-h5 file if one was given (made with
        # its own states, e.g. someone else's), otherwise states.yaml's
        if self.provided_assign is not None and os.path.isfile(self.provided_assign):
            with h5py.File(self.provided_assign, "r") as f:
                if "state_labels" not in f:
                    raise RuntimeError("assign-h5: {} has no state_labels; is it w_assign "
                                       "output made with states?".format(self.provided_assign))
                self.state_labels = [s.decode() if isinstance(s, bytes) else str(s)
                                     for s in f["state_labels"][:]]
        elif not os.path.isfile("states.yaml"):
            raise FileNotFoundError("states.yaml does not exist. Please run clustering analysis or assign macrostates")
        else:
            with open("states.yaml", "r") as file:
                states = yaml.load(file, Loader=Loader)
                self.state_labels = [state["label"] for state in states["states"]]

    def run(self):
        try:
            assign_stamp = self.ensure_assign()
            figs, direct = self.run_direct(assign_stamp)
            matrix_result = None
            if self.matrix:
                matrix_result, _ = self.run_matrix(assign_stamp)
            # kept for notebooks and tests: direct[(a, b)] is w_direct's last
            # window; matrix has the lumped transition matrix and MFPTs
            self.result = {"direct": direct, "matrix": matrix_result}
            self.write_report(direct, matrix_result)
            return figs
        finally:
            os.chdir(self.curr_path)

    # ------------------------------------------------------------------
    # direct (w_direct)
    # ------------------------------------------------------------------

    def run_direct(self, assign_stamp):
        command = [
            "w_direct",
            "all",
            "-W", "{}".format(self.h5file_path),
            "-a", self.assign_path,
            "--first-iter", "{}".format(self.first_iter),
            "--last-iter", "{}".format(self.last_iter),
            "--step-iter", "{}".format(self.step_iter),
            "-e", "cumulative",
        ]
        stamp = self.stamp("w_direct", assign=assign_stamp,
                           first_iter=int(self.first_iter),
                           last_iter=int(self.last_iter), step_iter=self.step_iter)
        # runs from the sim folder (needs west.cfg), writes direct.h5 there; moved here after
        self.run_westpa_tool("direct.h5", command, stamp, cwd=self.sim_path,
                             produced_at=os.path.join(self.sim_path, "direct.h5"),
                             log_path="direct_output.txt")

        # rate_evolution[window][from_state][to_state]: one record per
        # evaluation window, with fields iter_start, iter_stop, expected,
        # ci_lbound, ci_ubound (WESTPA's iter_block_ci_dtype). Read by name.
        with h5py.File("direct.h5", "r") as dirFile:
            rate_evolution = dirFile["rate_evolution"][:]

        figs = []
        final = {}
        for start_ind in range(len(self.state_labels)):
            for end_ind in range(len(self.state_labels)):
                if start_ind != end_ind:
                    start_state_label = self.state_labels[start_ind]
                    end_state_label = self.state_labels[end_ind]
                    rates = rate_evolution[:, start_ind, end_ind]
                    final[(start_ind, end_ind)] = rates[-1]
                    # each window's estimate uses iterations up to iter_stop
                    iters = rates["iter_stop"]
                    f, ax = plt.subplots(figsize=(8,6))
                    ax.plot(iters, rates["expected"], c='b')
                    ax.fill_between(iters, rates["ci_lbound"], rates["ci_ubound"], color='b', alpha=0.15)
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Transition Rate $\\tau^{-1}$')
                    ax.set_title(f'{start_state_label}$\\rightarrow${end_state_label}')
                    f.savefig("rate_{}_to_{}".format(start_state_label,end_state_label))
                    figs.append(f)
        return figs, final

    # ------------------------------------------------------------------
    # matrix (from assign.h5)
    # ------------------------------------------------------------------

    def run_matrix(self, assign_stamp):
        """The transition matrix from assign.h5 (each segment's first and last
        bin, weighted) and its MFPTs; see matrix_mfpts."""
        with h5py.File(self.assign_path, "r") as af:
            nbins = int(af.attrs["nbins"])
        labels, label_by_index = self.load_bin_state_assignment(self.assign_path, nbins)
        print("Building the transition matrix from {} (iterations {}-{})...".format(
            os.path.basename(self.assign_path), self.first_iter, self.last_iter))
        flux = markov.iter_bin_flux(self.assign_path, self.west_path, self.first_iter, self.last_iter)
        return self.matrix_mfpts(flux, labels, label_by_index)

    @staticmethod
    def load_bin_state_assignment(assign_h5_path, nbins):
        """Each bin's state from w_assign's state_map (-1: no state), and the labels."""
        with h5py.File(assign_h5_path, "r") as af:
            labels_raw = [s.decode() if isinstance(s, bytes) else s for s in af["state_labels"][:]]
            state_map = af["state_map"][:nbins]
        n = len(labels_raw)
        labels = np.where((state_map >= 0) & (state_map < n), state_map, -1).astype(int)
        return labels, dict(enumerate(labels_raw))

    def matrix_mfpts(self, flux, labels, label_by_index):
        """
        `flux` yields (n_iter, bin-to-bin weighted transitions of that
        iteration). Cumulative estimates every step-iter iterations from
        first_iter (and at last_iter): the transition matrix over the active
        bins, lumped into states, and MFPTs between every ordered pair of
        states (real time). Plots their evolution; returns the final estimates
        with a confidence interval from the spread of the last evaluations.
        """
        n = len(label_by_index)
        pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
        step = max(1, int(self.step_iter))
        history = {pr: ([], []) for pr in pairs}
        C = None
        final = None
        for i, C_i in flux:
            C = C_i.copy() if C is None else C + C_i
            if (i - int(self.first_iter)) % step and i != int(self.last_iter):
                continue
            act = markov.active_set(C)
            if len(act) < 2:          # no back-and-forth between bins yet
                continue
            T = markov.row_normalize(C[np.ix_(act, act)])
            pi = markov.stationary(T)
            lab = labels[act]
            for a, b in pairs:
                m = markov.mfpt(T, pi, lab == a, lab == b)
                if np.isfinite(m):
                    history[(a, b)][0].append(i)
                    history[(a, b)][1].append(m * self.tau)
            final = (act, T, pi, lab)
        if final is None:
            raise RuntimeError("network: in iterations {}-{} the walkers never moved back and forth "
                               "between bins, so there is no transition matrix to estimate (run "
                               "longer, or set matrix: false)".format(self.first_iter, self.last_iter))
        act, T, pi, lab = final
        P, p, nb = markov.coarse_grain(T, pi, lab, n)

        result = {"labels": [label_by_index[k] for k in range(n)], "state_prob": p,
                  "bins_per_state": nb, "transition_matrix": P, "mfpt": {}, "active_bins": len(act)}
        for a, b in pairs:
            its, ms = history[(a, b)]
            if not ms:
                continue
            window = np.array(ms[-min(40, len(ms)):])
            sem = window.std() / np.sqrt(len(window))
            result["mfpt"][(a, b)] = (ms[-1], ms[-1] - 1.96 * sem, ms[-1] + 1.96 * sem)
        self.plot_mfpt_evolution(history, label_by_index)
        return result, None

    def plot_mfpt_evolution(self, history, label_by_index):
        pairs = [pr for pr in history if history[pr][1]]
        if not pairs:
            return
        fig, axes = plt.subplots(len(pairs), 1, figsize=(8, 3.2 * len(pairs)), squeeze=False)
        for ax, (a, b) in zip(axes[:, 0], pairs):
            its, ms = history[(a, b)]
            ax.plot(its, ms, "b-", linewidth=1.5)
            ax.set_yscale("log")
            ax.set_title("{} -> {} (transition matrix)".format(label_by_index[a], label_by_index[b]))
            ax.set_xlabel("Iteration (cumulative from {})".format(self.first_iter))
            ax.set_ylabel("MFPT (tau = {})".format(self.tau))
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("mfpt_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------

    def write_report(self, direct, matrix):
        names = self.state_labels
        L = ["Network analysis: iterations {}-{}, tau = {} (times are WE iterations * tau)".format(
            self.first_iter, self.last_iter, self.tau), ""]
        if matrix is not None:
            n = len(matrix["labels"])
            L += ["Transition matrix (lag tau, from assign.h5), {} active bins".format(matrix["active_bins"]),
                  "{:>12}  {:>14}  {:>8}".format("State", "Stat. prob.", "Bins"), "-" * 38]
            L += ["{:>12}  {:>14.4g}  {:>8d}".format(matrix["labels"][k], matrix["state_prob"][k],
                                                   matrix["bins_per_state"][k]) for k in range(n)]
            L += ["", "State-to-state transition probabilities per WE iteration (row: from):",
                  " " * 12 + "".join("{:>11}".format(s) for s in matrix["labels"])]
            L += ["{:>12}".format(matrix["labels"][i]) + "".join(
                "{:>11.4g}".format(matrix["transition_matrix"][i, j]) for j in range(n)) for i in range(n)]
            L.append("")
        L += ["{:>24}  {:>12}  {:>25}  {:>12}".format("Transition", "MFPT direct", "MFPT matrix (95% CI)", "rate direct"),
              "-" * 80]
        for a in range(len(names)):
            for b in range(len(names)):
                if a == b:
                    continue
                key = "{} -> {}".format(names[a], names[b])
                rec = direct.get((a, b))
                k = float(rec["expected"]) if rec is not None else float("nan")
                m_direct = self.tau / k if k > 0 else float("inf")
                mm = matrix["mfpt"].get((a, b)) if matrix is not None else None
                m_matrix = "{:.4g} ({:.3g}, {:.3g})".format(*mm) if mm else "-"
                L.append("{:>24}  {:>12.4g}  {:>25}  {:>12.4g}".format(key, m_direct, m_matrix, k))
        L += ["", "direct: w_direct (history-labelled flux; MFPT = tau / rate, rate per tau).",
              "matrix: first passage from the transition matrix at lag tau, stationary-weighted over the",
              "        source state's bins. Precise rates of one transition: use a run targeted at it."]
        with open("network.txt", "w") as f:
            f.write("\n".join(L) + "\n")
        print("\n".join(L))
