import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
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
    Finds macrostates and writes them to states.yaml, for `network` (and any
    other tool that reads states.yaml). Two methods:

      dbscan  density: DBSCAN on the densest cells of the progress-coordinate
              histogram (pdist.h5). Options: threshold (percentile of density
              a cell needs), min-samples, eps.
      gpcca   dynamics: GPCCA+ on the run's bin-to-bin transition matrix
              (lag tau, one WE iteration), flux-symmetrized so each state keeps
              its stationary probability. Every bin the chain moves through
              back and forth gets a state. Options: n-clusters (null: GPCCA+
              picks the number, from 2 to max-clusters, by crispness). Uses the
              run's own bins (a w_assign without states), or the assign-h5 file
              if one is given, e.g. assignments with someone else's bins.

    Either way a state lists coordinates; w_assign puts every bin containing
    one of them in that state. gpcca lists one coordinate per bin: a progress
    coordinate the run actually had in that bin.
    """

    def __init__(self, opts):
        # gpcca runs w_assign, which opens west.h5 for writing while this
        # analysis holds it open to read: no file locking (as network does),
        # set before the base class opens it and inherited by the WESTPA tools
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        super().__init__(opts)
        self.method = str(self._getd(opts, "method", default="dbscan", required=False)).lower()
        if self.method not in METHODS:
            raise ValueError("cluster: method must be one of {}, got {!r}".format(
                ", ".join(METHODS), self.method))
        self.set_dims(self._getd(opts, "dimensions", required=False))
        # dbscan
        self.bins = self._getd(opts, "bins", default=30, required=False)
        self.threshold = self._getd(opts, "threshold", default=90, required=False)
        self.min_samples = self._getd(opts, "min-samples", default=2, required=False)
        self.eps = self._getd(opts, "eps", default=1.5, required=False)
        # gpcca
        self.n_clusters = self._getd(opts, "n-clusters", default=None, required=False)
        self.max_clusters = int(self._getd(opts, "max-clusters", default=10, required=False))

    def run(self):
        try:
            if self.method == "gpcca":
                return self.run_gpcca()
            return self.run_dbscan()
        finally:
            os.chdir(self.curr_path)

    # ------------------------------------------------------------------
    # dbscan
    # ------------------------------------------------------------------

    def run_dbscan(self):
        datFile = self.ensure_pdist()
        # pdist.h5 already covers exactly first_iter..last_iter, so use every row
        # (indexing it by iteration number again would double-offset the range)
        Hists = datFile["histograms"][:].mean(axis=0)

        # Figure out which bins are contained in high density regions
        density_threshold = np.percentile(Hists, self.threshold)
        high_density_bins = Hists > density_threshold
        dense_coords = np.column_stack(np.where(high_density_bins))

        # Run DBSCAN to cluster bins
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(dense_coords)
        cluster_labels = dbscan.labels_
        final_cluster_grid = np.full(Hists.shape,-1)
        for coord, label in zip(dense_coords, cluster_labels):
            final_cluster_grid[tuple(coord)] = label

        print("plotting macrostates")
        figs = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for jj in range(self.dims):
            for ii in range(jj+1,self.dims):
                # each dimension can have its own number of bins (see pdist_bin_edges)
                label_grid = [[set() for _ in range(final_cluster_grid.shape[ii])]
                              for _ in range(final_cluster_grid.shape[jj])]
                for index in np.ndindex(final_cluster_grid.shape):
                    x_ind = index[jj]
                    y_ind = index[ii]
                    label = final_cluster_grid[index]
                    if label != -1:
                        label_grid[x_ind][y_ind].add(label)
                label_grid = [[sorted(list(cell)) for cell in row] for row in label_grid]

                x_mid = datFile['midpoints_{}'.format(jj)][:]
                y_mid = datFile['midpoints_{}'.format(ii)][:]

                # Create a scatterplot displaying macrostates
                f, ax = plt.subplots(figsize=(8,6))
                for ix,x in enumerate(x_mid):
                    for iy,y in enumerate(y_mid):
                        label = label_grid[ix][iy]
                        if label == []:
                            s = 3
                            ax.scatter(x,y,marker='o',s=s,c='k')
                        else:
                            s = 80
                            for l in label:
                                ax.scatter(x,y,marker=f'${l}$',s=s,c=colors[l % len(colors)],alpha=0.5)
                ax.set_xlabel(self.names[jj])
                ax.set_ylabel(self.names[ii])
                f.savefig('cluster_{}_{}.png'.format(self.names[jj],self.names[ii]))
                figs.append(f)

        print("printing macrostates")
        # Each state's coords are the midpoints of its bins, as plain floats,
        # written by a YAML library. (Formatting NumPy numbers as text gives
        # "np.float64(0.5)" under NumPy 2, which YAML reads back as a string,
        # so w_assign couldn't use the file.)
        midpoints = [datFile['midpoints_{}'.format(d)][:] for d in range(final_cluster_grid.ndim)]
        states = []
        for label in range(int(final_cluster_grid.max()) + 1):
            coords = [[float(midpoints[d][i]) for d, i in enumerate(index)]
                      for index in zip(*np.where(final_cluster_grid == label))]
            states.append({"label": "state{}".format(label), "coords": coords})
        self.write_states_yaml(states)
        return figs

    # ------------------------------------------------------------------
    # gpcca
    # ------------------------------------------------------------------

    def ensure_bin_assign(self):
        """
        Bin assignments of every segment point, without states: the assign-h5
        file if one was given (with or without states), otherwise a w_assign
        of the whole run with its own bins, cached as assign_bins.h5. Returns
        its path.
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

    def run_gpcca(self):
        assign_path = self.ensure_bin_assign()
        print("Counting bin-to-bin transitions in iterations {}-{}...".format(
            self.first_iter, self.last_iter))
        C, rep, visited = markov.bin_flux_from_assignments(
            assign_path, self.west_path, self.first_iter, self.last_iter)
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

        figs = self.plot_bin_states(rep[act], labels, g["pi"], names)
        return figs

    def plot_bin_states(self, points, labels, pi, names):
        """
        Each bin as the progress coordinate representing it, coloured by state
        and sized by its stationary probability (log scale), in every pair of
        dimensions. Works for any binning scheme.
        """
        figs = []
        cmap = plt.get_cmap("tab10")
        lp = np.log10(np.maximum(pi, 1e-300))
        size = 8 + 60 * (lp - lp.min()) / max(np.ptp(lp), 1e-12)
        for jj in range(self.dims):
            for ii in range(jj + 1, self.dims):
                f, ax = plt.subplots(figsize=(8, 6))
                for k, name in enumerate(names):
                    sel = labels == k
                    ax.scatter(points[sel, jj], points[sel, ii], s=size[sel], color=cmap(k % 10),
                               edgecolors="k", linewidths=0.3, alpha=0.85,
                               label="{} ({} bins, p={:.2g})".format(name, int(sel.sum()), pi[sel].sum()))
                ax.set_xlabel(self.names[jj])
                ax.set_ylabel(self.names[ii])
                ax.set_title("GPCCA+ states (one point per bin; size ~ log stationary probability)")
                ax.legend(fontsize=7, loc="best")
                f.savefig("cluster_gpcca_{}_{}.png".format(self.names[jj], self.names[ii]),
                          dpi=150, bbox_inches="tight")
                figs.append(f)
        return figs

    @staticmethod
    def write_states_yaml(states, path="states.yaml"):
        with open(path, "w") as file:
            yaml.safe_dump({"states": states}, file, sort_keys=False, default_flow_style=None)
