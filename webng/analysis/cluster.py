import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from webng.analysis.analysis import weAnalysis
from webng.analysis import markov

# Hacky way to ignore warnings, in particular pyemma insists on Python3
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)

METHODS = ("dbscan", "gpcca")


class weCluster(weAnalysis):
    """
    Finds macrostates among the run's WE bins and writes them to states.yaml,
    for `network` (and any other tool that reads states.yaml). Two methods:

      dbscan  where the walkers spend their time: keep the most probable bins
              until they hold `probability-mass` of the probability (bins'
              average probability over the iterations), then group the kept
              bins that lie close together in progress-coordinate space with
              DBSCAN. Distances are measured after scaling every coordinate to
              [0, 1] over the range the run covered (after log10(x + 1) with
              `log-scale: true`), so `eps` is a fraction of that range, or
              chosen automatically (null): twice the median distance from a
              kept bin to its nearest kept neighbour. `min-samples` counts
              bins, the bin itself included. Bins left out by the cutoff, or
              that DBSCAN calls noise, belong to no state. Rare states (little
              probability) are cut first; for those use gpcca.
      gpcca   how the walkers move: GPCCA+ on the bin-to-bin transition matrix
              (lag tau, one WE iteration), flux-symmetrized so each state
              keeps its stationary probability. Every bin the walkers move
              back and forth through gets a state, rare ones included.
              `n-clusters`: the number of states, or null to let GPCCA+ choose
              between 2 and `max-clusters` by crispness. Also writes
              cluster_gpcca.txt (state probabilities and transition
              probabilities between states).

    Both use the run's own bins (a w_assign of the run, cached as
    assign_bins.h5) or the analyses' assign-h5 file if one is given, e.g.
    assignments with someone else's bins. Each state lists one coordinate per
    bin: a progress coordinate the run actually had in that bin, so w_assign
    puts exactly those bins in the state.
    """

    def __init__(self, opts):
        # w_assign opens west.h5 for writing while this analysis holds it open
        # to read: no file locking (as network does), set before the base
        # class opens it and inherited by the WESTPA tools
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        super().__init__(opts)
        self.method = str(self._getd(opts, "method", default="dbscan", required=False)).lower()
        if self.method not in METHODS:
            raise ValueError("cluster: method must be one of {}, got {!r}".format(
                ", ".join(METHODS), self.method))
        self.set_dims(self._getd(opts, "dimensions", required=False))
        # dbscan
        if opts.get("threshold") is not None:
            raise ValueError("cluster: 'threshold' (a percentile of histogram density) has been "
                             "replaced by 'probability-mass': keep the most probable bins until "
                             "they hold this fraction of the probability, e.g. 0.9")
        self.probability_mass = float(self._getd(opts, "probability-mass", default=0.9, required=False))
        if not 0 < self.probability_mass <= 1:
            raise ValueError("cluster: probability-mass must be in (0, 1], got {}".format(self.probability_mass))
        self.eps = self._getd(opts, "eps", default=None, required=False)
        if self.eps is not None and not 0 < float(self.eps) <= 1:
            raise ValueError("cluster: eps is a fraction of each progress coordinate's range, in "
                             "(0, 1] (or null to choose it automatically), got {}. (It used to be "
                             "a number of histogram cells.)".format(self.eps))
        self.min_samples = int(self._getd(opts, "min-samples", default=2, required=False))
        self.log_scale = bool(self._getd(opts, "log-scale", default=False, required=False))
        # gpcca
        self.n_clusters = self._getd(opts, "n-clusters", default=None, required=False)
        self.max_clusters = int(self._getd(opts, "max-clusters", default=10, required=False))

    def run(self):
        try:
            assign_path = self.ensure_bin_assign()
            print("Counting bin-to-bin transitions in iterations {}-{}...".format(
                self.first_iter, self.last_iter))
            C, rep, visited = markov.bin_flux_from_assignments(
                assign_path, self.west_path, self.first_iter, self.last_iter)
            if self.method == "gpcca":
                return self.run_gpcca(C, rep, visited)
            return self.run_dbscan(C, rep, visited)
        finally:
            os.chdir(self.curr_path)

    # ------------------------------------------------------------------
    # bin assignments, shared by both methods
    # ------------------------------------------------------------------

    def ensure_bin_assign(self):
        """
        Bin assignments of every segment point: the assign-h5 file if one was
        given (with or without states), otherwise a w_assign of the whole run
        with its own bins, cached as assign_bins.h5. Returns its path.
        """
        if self.provided_assign is not None:
            if not os.path.isfile(self.provided_assign):
                raise RuntimeError("assign-h5: {} does not exist".format(self.provided_assign))
            print("Using {} for bin assignments (analyses: assign-h5)".format(self.provided_assign))
            return self.provided_assign
        path = os.path.abspath("assign_bins.h5")

        def command():
            # w_assign without states crashes at its very end in WESTPA 2022.16
            # (it labels a 'statelabels' dataset it never made), so give it two
            # placeholder states: two points the run visited, valid for any
            # binning. Only the bin assignments are read; states don't change them.
            states = os.path.abspath("assign_bins_placeholder_states.yaml")
            with self._open_west() as f:
                first = f["iterations/iter_{:08d}".format(1)]["pcoord"][0, 0]
                last = f["iterations/iter_{:08d}".format(int(self.last_iter))]["pcoord"][-1, -1]
            self.write_states_yaml([{"label": "placeholder_a", "coords": [[float(v) for v in first]]},
                                    {"label": "placeholder_b", "coords": [[float(v) for v in last]]}],
                                   states)
            return (["w_assign", "-W", self.west_path, "--states-from-file", states, "-o", path]
                    + self.work_manager_args())
        # runs from the sim folder (w_assign needs its west.cfg / system.py)
        self.run_westpa_tool("assign_bins.h5", command, self.stamp("w_assign bins"),
                             cwd=self.sim_path)
        return path

    def _open_west(self):
        import h5py
        return h5py.File(self.west_path, "r")

    # ------------------------------------------------------------------
    # dbscan
    # ------------------------------------------------------------------

    def scaled_points(self, points, reference):
        """Points in the coordinates DBSCAN measures distances in: log10(x + 1)
        first with log-scale, then each dimension scaled to [0, 1] over
        `reference` (every visited bin's point)."""
        def tr(x):
            x = np.asarray(x, dtype=float)
            return np.log10(np.maximum(x, 0) + 1.0) if self.log_scale else x
        ref = tr(reference)
        lo, span = ref.min(axis=0), np.ptp(ref, axis=0)
        span[span == 0] = 1.0
        return (tr(points) - lo) / span

    @staticmethod
    def auto_eps(scaled):
        """Twice the median distance from a point to its nearest other point."""
        if len(scaled) < 2:
            return 1.0
        d = cKDTree(scaled).query(scaled, k=2)[0][:, 1]
        d = d[d > 0]
        return float(2 * np.median(d)) if len(d) else 1.0

    def run_dbscan(self, C, rep, visited):
        n_iter = int(self.last_iter) - int(self.first_iter) + 1
        p = C.sum(axis=0) / n_iter              # each bin's average probability (weight ending there)
        bins = np.nonzero(visited & (p > 0))[0]
        if len(bins) == 0:
            raise RuntimeError("cluster: no walker ended a segment in iterations {}-{}".format(
                self.first_iter, self.last_iter))
        # the most probable bins until they hold probability-mass of the probability
        order = bins[np.argsort(-p[bins], kind="stable")]
        cum = np.cumsum(p[order]) / p[bins].sum()
        core = order[: int(np.searchsorted(cum, self.probability_mass - 1e-12) + 1)]

        X = self.scaled_points(rep[core], rep[bins])
        eps = float(self.eps) if self.eps is not None else self.auto_eps(X)
        raw = DBSCAN(eps=eps, min_samples=self.min_samples).fit(X).labels_
        # states numbered by decreasing probability; noise (-1) is no state
        clusters = [c for c in np.unique(raw) if c >= 0]
        clusters.sort(key=lambda c: -p[core[raw == c]].sum())
        labels = np.full(len(core), -1)
        for k, c in enumerate(clusters):
            labels[raw == c] = k
        names = ["state{}".format(k) for k in range(len(clusters))]
        states = [{"label": names[k], "coords": [[float(v) for v in rep[b]] for b in core[labels == k]]}
                  for k in range(len(clusters))]
        self.write_states_yaml(states)

        total = p[bins].sum()
        lines = ["DBSCAN on the most probable bins (iterations {}-{})".format(self.first_iter, self.last_iter),
                 "kept {} of {} bins, holding {:.3f} of the probability (probability-mass {})".format(
                     len(core), len(bins), p[core].sum() / total, self.probability_mass),
                 "eps {:.4g} ({}) of each coordinate's range{}, min-samples {}; {} states, {} kept bins as noise".format(
                     eps, "automatic" if self.eps is None else "set", " after log10(x + 1)" if self.log_scale else "",
                     self.min_samples, len(clusters), int((labels == -1).sum())),
                 "",
                 "{:>8}  {:>14}  {:>8}".format("State", "Probability", "Bins"),
                 "-" * 34]
        lines += ["{:>8}  {:>14.4g}  {:>8d}".format(names[k], p[core[labels == k]].sum() / total,
                                                  int((labels == k).sum())) for k in range(len(clusters))]
        lines += ["", "MFPTs between these states: run the network analysis on this states.yaml."]
        with open("cluster_dbscan.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))

        # every visited bin in the plot; bins in no state in grey
        all_labels = np.full(len(bins), -1)
        pos = {b: i for i, b in enumerate(bins)}
        for b, l in zip(core, labels):
            all_labels[pos[b]] = l
        return self.plot_bin_states(rep[bins], all_labels, p[bins] / total, names, "dbscan",
                                    "DBSCAN states (one point per bin; size ~ log probability)")

    # ------------------------------------------------------------------
    # gpcca
    # ------------------------------------------------------------------

    def run_gpcca(self, C, rep, visited):
        print("Running GPCCA+ ({})...".format(
            "{} clusters".format(self.n_clusters) if self.n_clusters is not None
            else "number of clusters by crispness, 2 to {}".format(self.max_clusters)))
        g = markov.gpcca_states(C, self.n_clusters, self.max_clusters)
        act, labels, n = g["active"], g["labels"], g["n_states"]
        P, p, nb = markov.coarse_grain(g["T"], g["pi"], labels, n)
        names = ["state{}".format(k) for k in range(n)]

        # states.yaml: each state lists one coordinate inside each of its bins
        states = [{"label": names[k],
                   "coords": [[float(v) for v in rep[b]] for b in act[labels == k] if visited[b]]}
                  for k in range(n)]
        self.write_states_yaml(states)

        lines = ["GPCCA+ on the bin-to-bin transition matrix (lag tau, one WE iteration)",
                 "iterations {}-{}; {} of {} bins in the active set; {} states; crispness {:.4f}".format(
                     self.first_iter, self.last_iter, len(act), len(C), n, g["crispness"]),
                 "",
                 "{:>8}  {:>14}  {:>8}".format("State", "Stat. prob.", "Bins"),
                 "-" * 34]
        lines += ["{:>8}  {:>14.4g}  {:>8d}".format(names[k], p[k], nb[k]) for k in range(n)]
        lines += ["", "State-to-state transition probabilities per WE iteration (row: from):",
                  " " * 8 + "".join("{:>11}".format(s) for s in names)]
        lines += ["{:>8}".format(names[i]) + "".join("{:>11.4g}".format(P[i, j]) for j in range(n))
                  for i in range(n)]
        lines += ["", "MFPTs between these states: run the network analysis on this states.yaml."]
        with open("cluster_gpcca.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))

        return self.plot_bin_states(rep[act], labels, g["pi"], names, "gpcca",
                                    "GPCCA+ states (one point per bin; size ~ log stationary probability)")

    # ------------------------------------------------------------------
    # shared output
    # ------------------------------------------------------------------

    def plot_bin_states(self, points, labels, prob, names, method, title):
        """
        Each bin as the progress coordinate representing it, coloured by state
        (grey: no state) and sized by its probability (log scale), in every
        pair of dimensions. Works for any binning scheme.
        """
        figs = []
        cmap = plt.get_cmap("tab10")
        lp = np.log10(np.maximum(prob, 1e-300))
        size = 8 + 60 * (lp - lp.min()) / max(np.ptp(lp), 1e-12)
        groups = [(-1, "no state", "lightgrey")] + [(k, n, cmap(k % 10)) for k, n in enumerate(names)]
        for jj in range(self.dims):
            for ii in range(jj + 1, self.dims):
                f, ax = plt.subplots(figsize=(8, 6))
                for k, name, color in groups:
                    sel = labels == k
                    if not sel.any():
                        continue
                    ax.scatter(points[sel, jj], points[sel, ii], s=size[sel], color=color,
                               edgecolors="k", linewidths=0.3, alpha=0.85,
                               label="{} ({} bins, p={:.2g})".format(name, int(sel.sum()), prob[sel].sum()))
                ax.set_xlabel(self.names[jj])
                ax.set_ylabel(self.names[ii])
                ax.set_title(title)
                ax.legend(fontsize=7, loc="best")
                f.savefig("cluster_{}_{}_{}.png".format(method, self.names[jj], self.names[ii]),
                          dpi=150, bbox_inches="tight")
                figs.append(f)
        return figs

    @staticmethod
    def write_states_yaml(states, path="states.yaml"):
        with open(path, "w") as file:
            yaml.safe_dump({"states": states}, file, sort_keys=False, default_flow_style=None)
