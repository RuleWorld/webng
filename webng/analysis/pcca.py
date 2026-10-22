import os, sys, pickle, shutil
import subprocess as sbpc
import numpy as np
import h5py
import yaml
from yaml import Loader
from scipy.sparse import coo_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from webng.analysis.analysis import weAnalysis

import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)


class wePCCA(weAnalysis):
    """
    PCCA+/MFPT analysis: builds a per-bin transition matrix from a
    w_reweight run, computes mean first-passage times (MFPT) between
    macrostates, and runs an unsupervised GPCCA+ coarse-graining as a
    diagnostic.

    Ported and generalized from pcca_master.py (a validation script for
    the Tse et al. 2018 8-gene pluripotency network) by removing every
    hardcoded, system-specific piece:
      - genes/pcoords, tau, and sim paths now come from the same shared
        opts every other webng analysis tool receives (see
        weAnalysis.run() in weConvert.py)
      - macrostate labels come from states.yaml (the same convention
        `cluster` writes and `network` reads) via w_assign, instead of a
        hardcoded rectangular decision tree tied to specific gene
        thresholds. How states.yaml gets made (DBSCAN via `cluster`, or
        by hand) is not this tool's concern.
      - n-clusters/step-iter are now opts with defaults instead of
        module-level constants; burn-in is just first-iter from the
        overarching analyses block (no separate concept needed)
      - PAPER_VALS (Tse et al. reference numbers used for validation)
        removed entirely -- comparison to a reference is the user's
        business, not a general tool's
      - GPCCA+ coarse-graining switched from deeptime's
        MarkovStateModel.pcca() to pygpcca.GPCCA -- already a webng
        dependency (declared in setup.py), never previously wired up
        anywhere in the codebase, and the reference implementation of
        the method itself

    Two outputs, kept intentionally separate rather than reconciled:
      1. MFPT/rates between the states.yaml-labeled macrostates -- the
         well-defined, general-purpose output. states.yaml's own labels
         are used directly; there is no attempt to auto-identify which
         numbered cluster "means" which biological state.
      2. An unsupervised GPCCA+ coarse-graining of the same transition
         matrix, reported as numbered clusters with crispness
         diagnostics -- useful for checking whether your states.yaml
         regions line up with what the dynamics actually look like, but
         never auto-mapped onto your state labels. That mapping is
         exactly the system-specific heuristic this generalization is
         trying to avoid.

    Units: MFPT values are reported as real time, computed as
    (expected number of WE iterations) * tau. This assumes `tau` is the
    real-world duration of one WE iteration (e.g. seconds) -- the result
    is in whatever unit tau itself is expressed in, NOT "tau^-1" (that
    notation describes a rate, which is what network.py/flux.py report;
    MFPT is a time, never a rate).
    """

    def __init__(self, opts):
        # read west.h5 without file locking (set before the base class opens it)
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        super().__init__(opts)

        # tau always lives in sampling_options -- this must be injected
        # by the caller (either the dispatcher's existing
        # pcca_dict["sampling_options"] = self.opts["sampling_options"],
        # or by hand if constructing opts directly, e.g. from a notebook).
        sampling_opts = self._getd(opts, "sampling_options", default=None, required=True)
        self.tau = self._getd(sampling_opts, "tau", default=10, required=True)

        # pcca-specific options, all optional with sensible defaults
        self.n_clusters = self._getd(opts, "n-clusters", default=None, required=False)
        self.step = self._getd(opts, "step-iter", default=5, required=False)
        self.mfpt_pairs = self._getd(opts, "mfpt-pairs", default=None, required=False)

        # movement-h5: a separate west.h5 to load Voronoi bin centers from
        # (e.g. an earlier adaptive-binning phase, if bins were fixed
        # before this transition-mode run started). Defaults to the same
        # west.h5 being analyzed -- covers the common case where bins
        # were never re-adapted mid-run.
        movement_h5_opt = self._getd(opts, "movement-h5", default=None, required=False)
        # (a path relative to the sim folder)
        self.movement_h5_path = (
            os.path.join(self.sim_path, movement_h5_opt)
            if movement_h5_opt is not None
            else self.west_path
        )

        # states.yaml: same convention `cluster` writes and `network`
        # reads. How it got made is not this tool's concern -- it just
        # needs one to exist.
        if not os.path.isfile("states.yaml"):
            raise FileNotFoundError(
                "states.yaml does not exist. Please run clustering "
                "analysis or assign macrostates by hand before running "
                "pcca."
            )
        with open("states.yaml", "r") as f:
            states = yaml.load(f, Loader=Loader)
        self.state_labels = [s["label"] for s in states["states"]]

        if self.n_clusters is None:
            self.n_clusters = len(self.state_labels)

        if self.mfpt_pairs is None:
            self.mfpt_pairs = [
                (src, tgt)
                for src in self.state_labels
                for tgt in self.state_labels
                if src != tgt
            ]

    # ------------------------------------------------------------------
    # Ported directly from pcca_master.py -- pure array/HDF5 math, not
    # tied to the 8-gene system.
    # ------------------------------------------------------------------

    def load_final_centers(self, west_h5_path):
        """
        Reads the final iteration's Voronoi bin mapper out of a west.h5
        and returns its bin centers.
        """
        with h5py.File(west_h5_path, "r") as f:
            iters = sorted(f["iterations"].keys())
            last_iter_key = iters[-1]
            binhash = f["iterations/{}".format(last_iter_key)].attrs["binhash"]
            index = f["bin_topologies/index"][()]
            idx = None
            for i, entry in enumerate(index):
                if entry["hash"].decode("ascii").rstrip("\x00") == binhash:
                    idx = i
                    break
            if idx is None:
                raise ValueError("binhash not found in {}".format(west_h5_path))
            pickle_len = index[idx]["pickle_len"]
            mapper_bytes = f["bin_topologies/pickles"][idx, :pickle_len].tobytes()
            mapper = pickle.loads(mapper_bytes)
            return np.array(mapper.centers, dtype=np.float32)

    def build_transition_matrix(self, tmat_h5, first_iter=None, last_iter=None):
        """
        Builds the averaged, row-normalized per-bin transition matrix
        from a w_reweight tmat.h5. Ported as-is from pcca_master.py.
        """
        with h5py.File(tmat_h5, "r") as tmh5:
            nrows = tmh5.attrs["nrows"]
            ncols = tmh5.attrs["ncols"]
            iter_start = tmh5.attrs["iter_start"] if first_iter is None else first_iter
            iter_stop = tmh5.attrs["iter_stop"] if last_iter is None else last_iter
            # NOTE: this "2" is NOT the number of macrostates -- it's
            # w_reweight's own internal row/col doubling convention in
            # its flux-matrix format, unrelated to states.yaml or
            # n_clusters. Do not confuse this with n_clusters; changing
            # it would misparse the HDF5 layout, not change how many
            # macrostates you get.
            nstates = 2
            nbins = nrows // nstates

            tm_sum = None
            n_iters_used = 0
            all_iter_tms = {}

            for i in range(iter_start, iter_stop):
                it_str = "iter_{:08d}".format(i)
                if it_str not in tmh5["iterations"]:
                    continue
                col = tmh5["iterations"][it_str]["cols"][:]
                row = tmh5["iterations"][it_str]["rows"][:]
                flux = tmh5["iterations"][it_str]["flux"][:]
                ctm = coo_matrix((flux, (row, col)), shape=(nrows, ncols)).toarray()

                tm_i = np.zeros((nbins, nbins), dtype=np.float64)
                for bi in range(nbins):
                    for bj in range(nbins):
                        tm_i[bi, bj] = ctm[
                            bi * nstates : (bi + 1) * nstates,
                            bj * nstates : (bj + 1) * nstates,
                        ].sum()

                all_iter_tms[i] = tm_i
                tm_sum = tm_i.copy() if tm_sum is None else tm_sum + tm_i
                n_iters_used += 1

        if n_iters_used == 0:
            raise RuntimeError(
                "No iterations found in {} between {} and {} -- check "
                "first-iter/last-iter.".format(tmat_h5, iter_start, iter_stop)
            )

        print("  Used {} iterations".format(n_iters_used))

        tm_avg = tm_sum / n_iters_used
        row_sums = tm_avg.sum(axis=1)
        nz = row_sums > 0
        tm_norm = tm_avg.copy()
        tm_norm[nz] /= row_sums[nz, np.newaxis]

        return tm_norm, all_iter_tms, nbins

    def stationary_prob(self, T_active, n_active):
        """Eigendecomposition-based stationary distribution."""
        try:
            ev, evec = np.linalg.eig(T_active.T)
            si = np.argmin(np.abs(ev - 1.0))
            sp = np.real(evec[:, si])
            return np.abs(sp) / np.abs(sp).sum()
        except Exception:
            return np.ones(n_active) / n_active

    def compute_mfpt_pair(
        self, T_mat, n_act, f2a, assignments, src_label, tgt_label, stat_probs
    ):
        """
        First-passage-time calculation for a single (src, tgt) pair, in
        units of WE iterations (NOT yet scaled by tau -- that scaling
        happens in compute_mfpt_evolution, once, right before reporting).
        """
        tgt_full = np.where(assignments == tgt_label)[0]
        tgt_active = np.array([f2a[b] for b in tgt_full if b in f2a])
        non_tgt = np.array([i for i in range(n_act) if i not in set(tgt_active)])
        if len(tgt_active) == 0 or len(non_tgt) == 0:
            return np.nan
        T_QQ = T_mat[np.ix_(non_tgt, non_tgt)]
        A = np.eye(len(non_tgt)) - T_QQ
        rhs = np.ones(len(non_tgt))
        try:
            mfpt_q = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            mfpt_q, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        mfpt_full = np.zeros(n_act)
        mfpt_full[non_tgt] = mfpt_q
        src_full = np.where(assignments == src_label)[0]
        src_active = np.array([f2a[b] for b in src_full if b in f2a])
        if len(src_active) == 0:
            return np.nan
        probs = stat_probs[src_active]
        probs = probs / probs.sum()
        return float(np.sum(probs * mfpt_full[src_active]))

    def compute_mfpt_evolution(
        self, all_iter_tms, assignments_full, pairs, out_dir, label_by_index=None
    ):
        """
        Cumulative MFPT evolution for each (src_label, tgt_label) pair in
        `pairs` (integer state indices -- see run()). `label_by_index`,
        if given, is used ONLY to build human-readable display strings
        (e.g. "SC->TE" instead of "0->1") -- it never affects the
        underlying math, which always operates on the integer indices.

        Reported MFPT values are real time (iterations * tau), NOT
        "tau^-1" -- that notation is for rates (see network.py/flux.py),
        and MFPT is a time, never a rate. If tau is in seconds, these
        values are in seconds.
        """

        def label_for(idx):
            if label_by_index is not None:
                return label_by_index.get(idx, str(idx))
            return str(idx)

        iters_sorted = sorted(all_iter_tms.keys())
        if not iters_sorted:
            return {}
        # burn-in is just first-iter from the overarching analyses block
        # (already resolved by set_iter_range in __init__) -- no separate
        # concept needed.
        burn_in = self.first_iter
        iter_stop_ci = max(iters_sorted) + 1
        eval_iters = list(range(burn_in, iter_stop_ci, self.step))

        pair_results = {
            "{}->{}".format(label_for(s), label_for(t)): {"iters": [], "mfpts": []}
            for s, t in pairs
        }

        tm_cumsum = None
        n_summed = 0

        for i in iters_sorted:
            if i < burn_in:
                continue
            tm_cumsum = (
                all_iter_tms[i].copy() if tm_cumsum is None else tm_cumsum + all_iter_tms[i]
            )
            n_summed += 1

            if i not in eval_iters and i != iters_sorted[-1]:
                continue

            T_ci = tm_cumsum / n_summed
            rs = T_ci.sum(axis=1)
            nz = rs > 0
            T_ci[nz] /= rs[nz, np.newaxis]

            nz_rows = np.where(T_ci.sum(axis=1) != 0)[0]
            T_active = T_ci[np.ix_(nz_rows, nz_rows)]
            n_act = len(nz_rows)
            f2a = {full: act for act, full in enumerate(nz_rows)}
            sp = self.stationary_prob(T_active, n_act)

            for src_label, tgt_label in pairs:
                key = "{}->{}".format(label_for(src_label), label_for(tgt_label))
                m = self.compute_mfpt_pair(
                    T_active, n_act, f2a, assignments_full, src_label, tgt_label, sp
                )
                if not np.isnan(m):
                    # m is in units of WE iterations; * tau converts to
                    # real time (see units note in the docstring above).
                    pair_results[key]["iters"].append(i)
                    pair_results[key]["mfpts"].append(m * self.tau)

        ci_results = {}
        lines = []
        lines.append(
            "{:>16}  {:>18}  {:>18}  {:>18}".format(
                "Transition",
                "MFPT (real time)",
                "CI lower",
                "CI upper",
            )
        )
        lines.append(
            "(real time = iterations * tau; tau = {} -- if tau is in "
            "seconds, these values are in seconds)".format(self.tau)
        )
        lines.append("-" * 75)
        for src_label, tgt_label in pairs:
            key = "{}->{}".format(label_for(src_label), label_for(tgt_label))
            mfpts = np.array(pair_results[key]["mfpts"])
            if len(mfpts) < 10:
                lines.append("{:>16}  insufficient data".format(key))
                continue
            window = mfpts[-min(40, len(mfpts)) :]
            final = mfpts[-1]
            sem = window.std() / np.sqrt(len(window))
            ci_lo = final - 1.96 * sem
            ci_hi = final + 1.96 * sem
            ci_results[key] = (final, ci_lo, ci_hi)
            lines.append(
                "{:>16}  {:>18.3e}  {:>18.3e}  {:>18.3e}".format(
                    key, final, ci_lo, ci_hi
                )
            )
        print()
        for line in lines:
            print("  " + line)

        fig, axes = plt.subplots(
            len(pairs), 1, figsize=(8, 4 * max(1, len(pairs))), squeeze=False
        )
        for ax_i, (src_label, tgt_label) in enumerate(pairs):
            key = "{}->{}".format(label_for(src_label), label_for(tgt_label))
            iters = np.array(pair_results[key]["iters"])
            mfpts = np.array(pair_results[key]["mfpts"])
            ax = axes[ax_i][0]
            if len(mfpts) == 0:
                ax.set_title("{} (no data)".format(key))
                continue
            ax.plot(iters, mfpts, "b-", linewidth=1.5, label="Cumulative avg MFPT")
            if key in ci_results:
                final, ci_lo, ci_hi = ci_results[key]
                ax.axhline(
                    final,
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                    label="Final: {:.2e}".format(final),
                )
                ax.axhspan(
                    ci_lo,
                    ci_hi,
                    alpha=0.15,
                    color="blue",
                    label="95% CI: [{:.2e}, {:.2e}]".format(ci_lo, ci_hi),
                )
            ax.set_xlabel("Iteration (cumulative avg from iter {})".format(burn_in))
            ax.set_ylabel("MFPT (real time; iterations * tau={})".format(self.tau))
            ax.set_title("{} MFPT Convergence".format(key))
            ax.set_yscale("log")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, "mfpt_evolution.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  Saved mfpt_evolution.png")
        return ci_results, lines

    def plot_voronoi(
        self,
        centers,
        assignments,
        label_map,
        out_dir,
        title_suffix="",
        filename="voronoi_macrostates.png",
    ):
        """
        2D projections of Voronoi bin centers colored by macrostate.
        """
        ndim = centers.shape[1]
        dim_names = [self.names.get(i, str(i)) for i in range(ndim)]
        pairs_to_plot = [(i, j) for i in range(ndim) for j in range(i + 1, ndim)]
        if not pairs_to_plot:
            return

        unique_labels = sorted(set(assignments[assignments != -999]))
        cmap = plt.get_cmap("tab10")
        palette = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}

        n_panels = len(pairs_to_plot)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for panel_i, (di, dj) in enumerate(pairs_to_plot):
            ax = axes[panel_i]
            xdata, ydata = centers[:, dj], centers[:, di]
            for lbl in unique_labels:
                mask = assignments == lbl
                if mask.sum() == 0:
                    continue
                name = label_map.get(lbl, str(lbl))
                ax.scatter(
                    xdata[mask],
                    ydata[mask],
                    s=50,
                    alpha=0.85,
                    color=palette[lbl],
                    label="{} (n={})".format(name, mask.sum()),
                    edgecolors="k",
                    linewidths=0.3,
                    zorder=3,
                )
            ax.set_xlabel(dim_names[dj], fontsize=10)
            ax.set_ylabel(dim_names[di], fontsize=10)
            ax.legend(fontsize=6, loc="best")

        for panel_i in range(n_panels, len(axes)):
            axes[panel_i].axis("off")

        fig.suptitle("Voronoi Centers by Macrostate {}".format(title_suffix), fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved {}".format(filename))

    # ------------------------------------------------------------------
    # Reads the bin->state assignment out of w_assign's own assign.h5,
    # using state_map -- a direct bin_idx -> state_idx lookup table
    # w_assign already computes once, rather than re-deriving it from
    # the per-segment trajectory data (which solves a different problem
    # entirely -- tracing individual walkers through time, not "which
    # state is this bin in").
    # ------------------------------------------------------------------
    def load_bin_state_assignment(self, assign_h5_path, nbins):
        with h5py.File(assign_h5_path, "r") as af:
            state_labels_raw = [
                s.decode() if isinstance(s, bytes) else s
                for s in af["state_labels"][:]
            ]
            # state_map: bin_idx -> state_idx, with any value outside
            # [0, n_states) meaning "no state assigned" (WESTPA's
            # standard catch-all for bins outside every defined state).
            state_map = af["state_map"][:nbins]

        n_states = len(state_labels_raw)
        assignments_full = np.full(nbins, -999, dtype=int)
        for i, state_idx in enumerate(state_map):
            if 0 <= state_idx < n_states:
                assignments_full[i] = state_idx

        label_by_index = {i: lbl for i, lbl in enumerate(state_labels_raw)}
        return assignments_full, label_by_index

    # ------------------------------------------------------------------
    # GPCCA+ diagnostic clustering -- switched from deeptime's
    # MarkovStateModel.pcca() to pygpcca.GPCCA. Produces its own numbered
    # clusters independent of states.yaml -- see the class docstring for
    # why cluster numbers are intentionally not auto-mapped onto your
    # state labels.
    # ------------------------------------------------------------------
    def run_gpcca_diagnostic(self, tm_norm, centers, out_dir):
        try:
            import pygpcca as gp
        except ImportError:
            print("  pygpcca not available -- skipping GPCCA+ diagnostic")
            return None, ["GPCCA+ diagnostic: pygpcca not available, skipped"]

        nonzero_rows_idx = np.where(tm_norm.sum(axis=1) != 0)[0]
        T_visited = tm_norm[np.ix_(nonzero_rows_idx, nonzero_rows_idx)]

        # Symmetrize to enforce detailed balance -- required for a
        # meaningful PCCA-style decomposition (kept from pcca_master.py).
        T_sym = (T_visited + T_visited.T) / 2.0
        rs = T_sym.sum(axis=1)
        rs[rs == 0] = 1.0
        T_sym = T_sym / rs[:, np.newaxis]

        print(
            "  Running GPCCA+ with n_clusters={} on {} bins...".format(
                self.n_clusters, len(nonzero_rows_idx)
            )
        )
        try:
            gpcca = gp.GPCCA(T_sym)
            # optimize(m) pins to exactly m clusters. The {"m_min":...,
            # "m_max":...} dict form is for auto-selecting the best m
            # over a real range (m_min < m_max) -- it rejects a
            # degenerate range where both ends are equal, which is what
            # happens if you pass n_clusters to both.
            gpcca.optimize(self.n_clusters)
            hard_assignments_visited = gpcca.macrostate_assignment
            crispness = gpcca.optimal_crispness
            print("  Crispness: {:.4f}".format(crispness))
        except Exception as e:
            msg = "GPCCA+ failed: {}".format(e)
            print("  " + msg)
            return None, [msg]

        nbins = tm_norm.shape[0]
        assignments_full = np.full(nbins, -999, dtype=int)
        for i, bin_idx in enumerate(nonzero_rows_idx):
            assignments_full[bin_idx] = hard_assignments_visited[i]

        label_map = {m: "C{}".format(m) for m in range(self.n_clusters)}

        cg_tm = gpcca.coarse_grained_transition_matrix
        cg_sp = gpcca.coarse_grained_stationary_probability

        lines = []
        lines.append(
            "GPCCA+ diagnostic clustering (n_clusters={}, crispness={:.4f}):".format(
                self.n_clusters, crispness
            )
        )
        lines.append("{:>8}  {:>14}  {:>10}".format("Cluster", "Stat. prob.", "N bins"))
        lines.append("-" * 36)
        for m in range(self.n_clusters):
            n_bins_in_cluster = int(np.sum(hard_assignments_visited == m))
            lines.append(
                "{:>8}  {:>14.4f}  {:>10d}".format(
                    label_map[m], cg_sp[m], n_bins_in_cluster
                )
            )

        lines.append("")
        lines.append("Coarse-grained transition matrix (probability per WE iteration):")
        header = " " * 8 + "".join(
            "{:>10}".format(label_map[m]) for m in range(self.n_clusters)
        )
        lines.append(header)
        for i in range(self.n_clusters):
            row_str = "{:>8}".format(label_map[i]) + "".join(
                "{:>10.4f}".format(cg_tm[i, j]) for j in range(self.n_clusters)
            )
            lines.append(row_str)

        print()
        for line in lines[1:]:  # skip the header line, already printed via "Crispness: ..." above
            print("  " + line)

        self.plot_voronoi(
            centers,
            assignments_full,
            label_map,
            out_dir,
            title_suffix="(GPCCA+ diagnostic)",
            filename="voronoi_gpcca_diagnostic.png",
        )

        return {
            "crispness": crispness,
            "assignments": assignments_full,
            "coarse_grained_transition_matrix": gpcca.coarse_grained_transition_matrix,
            "coarse_grained_stationary_probability": gpcca.coarse_grained_stationary_probability,
        }, lines

    def summarize_states_yaml_partition(self, tm_norm, assignments_full, label_by_index):
        """
        Prints the same two metrics run_gpcca_diagnostic prints for its
        own clusters -- stationary probability + bin count per state,
        and the coarse-grained transition matrix -- but for the
        states.yaml-labeled partition instead. This is the same
        coarse-graining operation GPCCA+ does internally (aggregate a
        bin-level transition matrix onto a hard partition, weighted by
        each bin's own stationary probability), just applied to your
        states.yaml assignment instead of GPCCA+'s own clustering, so
        the two are directly comparable using the same numbers.

        Bins not covered by any states.yaml label (assignment == -999)
        are excluded entirely -- their probability mass isn't folded
        into any state, by design (states.yaml only defines what it
        defines; this isn't trying to infer coverage for unlabeled
        bins).
        """
        nz_rows = np.where(tm_norm.sum(axis=1) != 0)[0]
        T_active = tm_norm[np.ix_(nz_rows, nz_rows)]
        n_active = len(nz_rows)
        sp = self.stationary_prob(T_active, n_active)

        state_indices = sorted(label_by_index.keys())
        n_states = len(state_indices)

        bins_per_state = {
            s: [
                act_idx
                for act_idx, full_idx in enumerate(nz_rows)
                if assignments_full[full_idx] == s
            ]
            for s in state_indices
        }

        cg_sp = np.zeros(n_states)
        cg_tm = np.zeros((n_states, n_states))
        bin_counts = np.zeros(n_states, dtype=int)

        for ii, si in enumerate(state_indices):
            bins_i = bins_per_state[si]
            bin_counts[ii] = len(bins_i)
            weight_i = sp[bins_i].sum() if bins_i else 0.0
            cg_sp[ii] = weight_i
            if weight_i == 0:
                continue
            for jj, sj in enumerate(state_indices):
                bins_j = bins_per_state[sj]
                if not bins_j:
                    continue
                flux_ij = sp[bins_i][:, np.newaxis] * T_active[np.ix_(bins_i, bins_j)]
                cg_tm[ii, jj] = flux_ij.sum() / weight_i

        lines = []
        lines.append(
            "states.yaml partition summary (same metrics as the GPCCA+ "
            "diagnostic below, for direct comparison):"
        )
        lines.append("{:>8}  {:>14}  {:>10}".format("State", "Stat. prob.", "N bins"))
        lines.append("-" * 36)
        for ii, si in enumerate(state_indices):
            lines.append(
                "{:>8}  {:>14.4f}  {:>10d}".format(
                    label_by_index[si], cg_sp[ii], bin_counts[ii]
                )
            )

        lines.append("")
        lines.append("Coarse-grained transition matrix (probability per WE iteration):")
        header = " " * 8 + "".join(
            "{:>10}".format(label_by_index[si]) for si in state_indices
        )
        lines.append(header)
        for ii, si in enumerate(state_indices):
            row_str = "{:>8}".format(label_by_index[si]) + "".join(
                "{:>10.4f}".format(cg_tm[ii, jj]) for jj in range(n_states)
            )
            lines.append(row_str)

        print()
        for line in lines:
            print("  " + line)

        return {
            "stationary_probability": cg_sp,
            "transition_matrix": cg_tm,
            "bin_counts": bin_counts,
            "state_order": [label_by_index[s] for s in state_indices],
        }, lines

    def run(self):
        assign_h5 = "assign.h5"
        tmat_h5 = "tmat.h5"

        assign_stamp = self.ensure_assign()

        # runs from the sim folder (w_reweight needs its west.cfg)
        command = [
            "w_reweight",
            "init",
            "-W", self.west_path,
            "-a", os.path.abspath(assign_h5),
            "-o", os.path.abspath(tmat_h5),
        ]
        self.run_westpa_tool(tmat_h5, command,
                             self.stamp("w_reweight init", assign=assign_stamp),
                             cwd=self.sim_path)

        print("Building transition matrix...")
        tm_norm, all_iter_tms, nbins = self.build_transition_matrix(
            tmat_h5, self.first_iter, self.last_iter
        )

        print("Loading Voronoi bin centers...")
        centers = self.load_final_centers(self.movement_h5_path)

        print("Loading states.yaml-based bin assignment from assign.h5...")
        assignments_full, label_by_index = self.load_bin_state_assignment(
            assign_h5, nbins
        )
        index_by_label = {lbl: i for i, lbl in label_by_index.items()}

        print("Computing MFPT between states.yaml-labeled macrostates...")
        pairs_as_indices = [
            (index_by_label[src], index_by_label[tgt])
            for src, tgt in self.mfpt_pairs
            if src in index_by_label and tgt in index_by_label
        ]
        ci_results, mfpt_lines = self.compute_mfpt_evolution(
            all_iter_tms, assignments_full, pairs_as_indices, ".", label_by_index
        )

        states_summary, states_lines = self.summarize_states_yaml_partition(
            tm_norm, assignments_full, label_by_index
        )

        self.plot_voronoi(
            centers,
            assignments_full,
            label_by_index,
            ".",
            title_suffix="(states.yaml)",
            filename="voronoi_states.png",
        )

        print("Running GPCCA+ diagnostic clustering...")
        gpcca_results, gpcca_lines = self.run_gpcca_diagnostic(tm_norm, centers, ".")

        # Results only -- no progress messages ("Building transition
        # matrix...") or save confirmations ("Saved X.png"), just the
        # actual tables each section already printed.
        report_lines = []
        report_lines.append("MFPT between states.yaml-labeled macrostates")
        report_lines.append("=" * 60)
        report_lines.extend(mfpt_lines)
        report_lines.append("")
        report_lines.append("states.yaml partition summary")
        report_lines.append("=" * 60)
        report_lines.extend(states_lines[1:])  # skip states_lines' own duplicate title
        report_lines.append("")
        report_lines.append("GPCCA+ diagnostic clustering")
        report_lines.append("=" * 60)
        report_lines.extend(gpcca_lines[1:] if gpcca_results is not None else gpcca_lines)

        with open("output_pcca.txt", "w") as f:
            f.write("\n".join(report_lines) + "\n")

        os.chdir(self.curr_path)
        return {
            "mfpt": ci_results,
            "states_summary": states_summary,
            "gpcca": gpcca_results,
        }