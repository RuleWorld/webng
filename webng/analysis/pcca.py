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
      - n-clusters/burn-in/step are now opts with defaults instead of
        module-level constants
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

    NOTE on verification: the pure array/math functions here (
    stationary_prob, compute_mfpt_pair, the transition-matrix
    normalization) were ported directly from pcca_master.py and
    additionally checked against synthetic data. The HDF5-reading code
    (build_transition_matrix, load_final_centers,
    load_bin_state_assignment) and the pygpcca call could not be run
    against a real WESTPA output or a real pygpcca install in the
    environment this was written in -- these are the parts most likely
    to need a small correction on first real run, and are commented
    accordingly at each site.
    """

    def __init__(self, opts):
        super().__init__(opts)
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        self.h5file_path = "west.h5"
        self.h5file = h5py.File(os.path.join("..", self.h5file_path), "r")

        # iterations
        self.first_iter = self._getd(opts, "first-iter", default=None, required=False)
        self.last_iter = self._getd(opts, "last-iter", default=None, required=False)
        self.first_iter, self.last_iter = self.set_iter_range(
            self.first_iter, self.last_iter
        )

        self.tau = self._getd(opts, "tau", default=10, required=True)

        # pcca-specific options, all optional with sensible defaults
        self.n_clusters = self._getd(opts, "n-clusters", default=None, required=False)
        self.burn_in = self._getd(opts, "burn-in", default=None, required=False)
        self.step = self._getd(opts, "step", default=5, required=False)
        self.mfpt_pairs = self._getd(opts, "mfpt-pairs", default=None, required=False)

        # movement-h5: a separate west.h5 to load Voronoi bin centers from
        # (e.g. an earlier adaptive-binning phase, if bins were fixed
        # before this transition-mode run started). Defaults to the same
        # west.h5 being analyzed -- covers the common case where bins
        # were never re-adapted mid-run.
        movement_h5_opt = self._getd(opts, "movement-h5", default=None, required=False)
        self.movement_h5_path = (
            os.path.join("..", movement_h5_opt)
            if movement_h5_opt is not None
            else os.path.join("..", self.h5file_path)
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

    def set_iter_range(self, first_iter, last_iter):
        if first_iter is None:
            first_iter = 1
        if last_iter is None:
            last_iter = self.h5file.attrs["west_current_iteration"] - 1
        return first_iter, last_iter

    # ------------------------------------------------------------------
    # Ported directly from pcca_master.py -- pure array/HDF5 math, not
    # tied to the 8-gene system.
    # ------------------------------------------------------------------

    def load_final_centers(self, west_h5_path):
        """
        Reads the final iteration's Voronoi bin mapper out of a west.h5
        and returns its bin centers. NOTE: could not test against a real
        west.h5 in the environment this was written in -- ported as-is
        from pcca_master.py, which Alex has already run successfully
        against real WESTPA output.
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
        """Eigendecomposition-based stationary distribution. Verified
        against synthetic transition matrices (see chat)."""
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
        """First-passage-time calculation for a single (src, tgt) pair.
        Verified against an analytically-solvable synthetic 2-state
        system (see chat)."""
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

    def compute_mfpt_evolution(self, all_iter_tms, assignments_full, pairs, out_dir):
        """
        Cumulative MFPT evolution for each (src_label, tgt_label) pair in
        `pairs`. Generalized from pcca_master.py's version, which only
        handled the hardcoded SC/TE/LN1 triple, to an arbitrary list of
        label pairs (labels here are the integer indices returned by
        load_bin_state_assignment, not the string labels themselves --
        see run()).
        """
        iters_sorted = sorted(all_iter_tms.keys())
        if not iters_sorted:
            return {}
        burn_in = (
            self.burn_in if self.burn_in is not None else int(0.2 * iters_sorted[-1])
        )
        iter_stop_ci = max(iters_sorted) + 1
        eval_iters = list(range(burn_in, iter_stop_ci, self.step))

        pair_results = {
            "{}->{}".format(s, t): {"iters": [], "mfpts": []} for s, t in pairs
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
                key = "{}->{}".format(src_label, tgt_label)
                m = self.compute_mfpt_pair(
                    T_active, n_act, f2a, assignments_full, src_label, tgt_label, sp
                )
                if not np.isnan(m):
                    pair_results[key]["iters"].append(i)
                    pair_results[key]["mfpts"].append(m * self.tau)

        ci_results = {}
        print(
            "\n  {:>16}  {:>14}  {:>14}  {:>14}".format(
                "Transition", "MFPT (tau^-1)", "CI lower", "CI upper"
            )
        )
        print("  " + "-" * 65)
        for src_label, tgt_label in pairs:
            key = "{}->{}".format(src_label, tgt_label)
            mfpts = np.array(pair_results[key]["mfpts"])
            if len(mfpts) < 10:
                print("  {:>16}  insufficient data".format(key))
                continue
            window = mfpts[-min(40, len(mfpts)) :]
            final = mfpts[-1]
            sem = window.std() / np.sqrt(len(window))
            ci_lo = final - 1.96 * sem
            ci_hi = final + 1.96 * sem
            ci_results[key] = (final, ci_lo, ci_hi)
            print(
                "  {:>16}  {:>14.3e}  {:>14.3e}  {:>14.3e}".format(
                    key, final, ci_lo, ci_hi
                )
            )

        fig, axes = plt.subplots(
            len(pairs), 1, figsize=(8, 4 * max(1, len(pairs))), squeeze=False
        )
        for ax_i, (src_label, tgt_label) in enumerate(pairs):
            key = "{}->{}".format(src_label, tgt_label)
            iters = np.array(pair_results[key]["iters"])
            mfpts = np.array(pair_results[key]["mfpts"])
            ax = axes[ax_i][0]
            if len(mfpts) == 0:
                ax.set_title("{}->{} (no data)".format(src_label, tgt_label))
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
            ax.set_ylabel("MFPT (tau^-1)")
            ax.set_title("{}->{} MFPT Convergence".format(src_label, tgt_label))
            ax.set_yscale("log")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, "mfpt_evolution.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  Saved mfpt_evolution.png")
        return ci_results

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
        Generalized from pcca_master.py's plot_voronoi (which hardcoded
        3 specific gene-pair panels) to plot every pair of dimensions
        using whatever pcoord names are available.
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
    # New code: reads the bin->state assignment out of w_assign's own
    # assign.h5, instead of pcca_master.py's bespoke geometric
    # classifier (which was hardcoded to the 8-gene system's specific
    # thresholds and isn't general). This reuses WESTPA's own state
    # membership logic -- the same thing states.yaml + w_assign already
    # do for `network` -- rather than re-implementing region containment
    # ourselves.
    #
    # NOTE: could not verify this against a real assign.h5 -- no h5py/
    # WESTPA available in the environment this was written in. The
    # dataset key names below (state_labels, bin_assignments) match
    # WESTPA's documented w_assign output format, but this is the one
    # piece of this file most likely to need a small correction on your
    # first real run. If the key names are wrong, the actual h5 file
    # will raise a clear KeyError naming what's missing, not silently
    # give a wrong answer -- check that error against `h5dump -n
    # assign.h5` or `h5py.File("assign.h5").keys()` if it happens.
    # ------------------------------------------------------------------
    def load_bin_state_assignment(self, assign_h5_path, nbins):
        with h5py.File(assign_h5_path, "r") as af:
            state_labels_raw = [
                s.decode() if isinstance(s, bytes) else s
                for s in af["state_labels"][:]
            ]
            # per-iteration, per-bin -> state index (or len(state_labels)
            # / -1 for "no state"). We only need one representative
            # bin->state map, so we take the most recent iteration.
            assignments_ds = af["bin_assignments"]
            last_iter_assignments = assignments_ds[-1][:nbins]
        assignments_full = np.full(nbins, -999, dtype=int)
        for i, state_idx in enumerate(last_iter_assignments):
            if 0 <= state_idx < len(state_labels_raw):
                assignments_full[i] = state_idx
        label_by_index = {i: lbl for i, lbl in enumerate(state_labels_raw)}
        return assignments_full, label_by_index

    # ------------------------------------------------------------------
    # GPCCA+ diagnostic clustering -- switched from deeptime's
    # MarkovStateModel.pcca() to pygpcca.GPCCA (already a webng
    # dependency, never previously wired up anywhere in the codebase).
    # Produces its own numbered clusters independent of states.yaml --
    # see the class docstring for why cluster numbers are intentionally
    # not auto-mapped onto your state labels.
    #
    # NOTE: could not run this against a real pygpcca install -- not
    # available in the environment this was written in, and no network
    # access to install it. Written against pygpcca's documented API
    # (fetched directly from its docs, not from memory), but this is the
    # other piece of this file most likely to need a small correction on
    # your first real run.
    # ------------------------------------------------------------------
    def run_gpcca_diagnostic(self, tm_norm, centers, out_dir):
        try:
            import pygpcca as gp
        except ImportError:
            print("  pygpcca not available -- skipping GPCCA+ diagnostic")
            return None

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
            gpcca.optimize({"m_min": self.n_clusters, "m_max": self.n_clusters})
            hard_assignments_visited = gpcca.macrostate_assignment
            crispness = gpcca.optimal_crispness
            print("  Crispness: {:.4f}".format(crispness))
        except Exception as e:
            print("  GPCCA+ failed: {}".format(e))
            return None

        nbins = tm_norm.shape[0]
        assignments_full = np.full(nbins, -999, dtype=int)
        for i, bin_idx in enumerate(nonzero_rows_idx):
            assignments_full[bin_idx] = hard_assignments_visited[i]

        label_map = {m: "C{}".format(m) for m in range(self.n_clusters)}
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
        }

    def run(self):
        assign_h5 = "assign.h5"
        tmat_h5 = "tmat.h5"

        if not os.path.isfile(assign_h5):
            print("assign.h5 does not exist. Running w_assign")
            command = [
                "w_assign",
                "-W",
                "{}".format(self.h5file_path),
                "--states-from-file",
                "./analysis/states.yaml",
                "-o",
                "./analysis/{}".format(assign_h5),
            ]
            proc = sbpc.Popen(command, cwd="../")
            proc.wait()

        if not os.path.isfile(tmat_h5):
            print("tmat.h5 does not exist. Running w_reweight")
            command = [
                "w_reweight",
                "init",
                "-W",
                "{}".format(self.h5file_path),
                "-a",
                "./analysis/{}".format(assign_h5),
                "-o",
                "./analysis/{}".format(tmat_h5),
            ]
            proc = sbpc.Popen(command, cwd="../")
            proc.wait()

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
        # NOTE: cwd is already self.work_path (weAnalysis.__init__
        # chdir'd there) -- matching every other analysis tool's
        # convention (average.py/evolution.py/flux.py all savefig with
        # a bare relative filename), out_dir here is "." not
        # self.work_path, or paths would double up.
        ci_results = self.compute_mfpt_evolution(
            all_iter_tms, assignments_full, pairs_as_indices, "."
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
        gpcca_results = self.run_gpcca_diagnostic(tm_norm, centers, ".")

        os.chdir(self.curr_path)
        return {"mfpt": ci_results, "gpcca": gpcca_results}